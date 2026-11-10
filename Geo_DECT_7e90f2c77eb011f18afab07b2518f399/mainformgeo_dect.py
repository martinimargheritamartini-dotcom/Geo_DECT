import os
import re
import csv
import sys
import gc
import json
import inspect
import traceback

import numpy as np
import tifffile as tf
from scipy import ndimage
from PyQt6 import QtCore, QtGui, QtWidgets

from OrsLibraries.libfunctions import LibFunctions
from OrsLibraries.workingcontext import WorkingContext
from OrsHelpers.layoutpropertieshelper import LayoutPropertiesHelper
from OrsHelpers.layoutHelper import LayoutHelper
from ORSServiceClass.windowclasses.orsabstractwindow import OrsAbstractWindow

from .ui_mainformgeo_dect import Ui_MainFormGeo_dect
from .geodect_core import (
    WATER_ELECTRON_DENSITY_CM3,
    compute_hu,
    compute_mu,
    compute_zeff,
    compute_rhoe,
    relative_to_absolute_rhoe,
    denoise_mu,
)

from .geodect_visualization import (
    create_geodect_rgb,
    create_zeff_rgb,
    create_rhoe_rgb,
    create_geodect_legend,
    create_zeff_colorbar,
    create_rhoe_colorbar,
    create_tick_values,
    ZEFF_PALETTE_RGB255,
    RHOE_PALETTE_RGB255,
)


class GeoDECTScatterRangeSelector(QtWidgets.QWidget):
    """
    Interactive 2D colorimetric range selector for the combined Z_eff /
    rho_e display.

    Shows the CIE Lab colour field over the FULL absolute stack range
    (fixed background, computed once) with a draggable/resizable
    rectangle overlay representing the currently selected Z_eff x rho_e
    sub-range -- "crop tool" style: drag the interior to move it, drag an
    edge or corner to resize it.

    This widget ONLY updates the rectangle overlay live (fast, purely
    visual, no recompute). It never triggers the expensive combined-volume
    re-encoding itself -- the owner must read get_selection() and act on
    it explicitly (e.g. from an "Apply" button), because re-encoding the
    whole combined volume for every intermediate drag frame would be far
    too slow.
    """

    selectionChanged = QtCore.pyqtSignal(float, float, float, float)

    _HANDLE_PIXELS = 9

    def __init__(
        self,
        zeff_full_minimum,
        zeff_full_maximum,
        rhoe_full_minimum,
        rhoe_full_maximum,
        zeff_selected_minimum,
        zeff_selected_maximum,
        rhoe_selected_minimum,
        rhoe_selected_maximum,
        rhoe_display_scale=1.0,
        parent=None,
    ):
        super().__init__(parent)

        self._zeff_full_minimum = float(zeff_full_minimum)
        self._zeff_full_maximum = float(zeff_full_maximum)
        self._rhoe_full_minimum = float(rhoe_full_minimum)
        self._rhoe_full_maximum = float(rhoe_full_maximum)
        # rho_e axis values are stored/displayed already divided by this
        # scale (e.g. 1e23), to keep numbers human-sized. Multiply back
        # by this before formatting tick labels, so the exponent shown
        # (e.g. "x10^23") is the true physical one, not "x10^0".
        self._rhoe_display_scale = float(rhoe_display_scale)

        self._selection = [
            float(zeff_selected_minimum),
            float(zeff_selected_maximum),
            float(rhoe_selected_minimum),
            float(rhoe_selected_maximum),
        ]

        self._margin_left = 76
        self._margin_right = 16
        self._margin_top = 12
        self._margin_bottom = 56

        self._background_pixmap = None
        self._drag_mode = None
        self._drag_start_pos = None
        self._drag_start_selection = None

        self.setMinimumSize(260, 260)
        self.setMouseTracking(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._points = []

    def get_selection(self):
        return tuple(self._selection)

    def add_point(self, zeff_value, rhoe_value):
        self._points.append((float(zeff_value), float(rhoe_value)))
        self.update()

    def clear_points(self):
        self._points = []
        self.update()

    def set_selection(
        self,
        zeff_minimum,
        zeff_maximum,
        rhoe_minimum,
        rhoe_maximum,
        emit_signal=False,
    ):
        self._selection = [
            float(zeff_minimum),
            float(zeff_maximum),
            float(rhoe_minimum),
            float(rhoe_maximum),
        ]
        self.update()
        if emit_signal:
            self.selectionChanged.emit(*self._selection)

    def _size_scale(self):
        """Scale factor relative to the small in-panel baseline size
        (260px) -- used to make margins/fonts genuinely bigger on the
        larger temporary instance built for "Enlarge / Save", not just
        the same small values stretched onto a bigger canvas."""
        return max(self.width(), self.height()) / 260.0

    def _grid_rect(self):
        scale = self._size_scale()
        margin_left = self._margin_left * scale
        margin_right = self._margin_right * scale
        margin_top = self._margin_top * scale
        margin_bottom = self._margin_bottom * scale
        return QtCore.QRectF(
            margin_left,
            margin_top,
            max(self.width() - margin_left - margin_right, 1),
            max(self.height() - margin_top - margin_bottom, 1),
        )

    def _ensure_background_pixmap(self):
        grid = self._grid_rect()
        width = max(int(grid.width()), 2)
        height = max(int(grid.height()), 2)
        if (
            self._background_pixmap is not None
            and self._background_pixmap.width() == width
            and self._background_pixmap.height() == height
        ):
            return

        rgb = create_geodect_legend(
            self._zeff_full_minimum,
            self._zeff_full_maximum,
            self._rhoe_full_minimum,
            self._rhoe_full_maximum,
            width=width,
            height=height,
        )
        self._background_pixmap = MainFormGeo_dect._numpy_rgb_to_pixmap(rgb)

    def _data_to_pixel(self, zeff_value, rhoe_value):
        grid = self._grid_rect()
        zeff_fraction = (zeff_value - self._zeff_full_minimum) / max(
            self._zeff_full_maximum - self._zeff_full_minimum, 1e-12
        )
        rhoe_fraction = (rhoe_value - self._rhoe_full_minimum) / max(
            self._rhoe_full_maximum - self._rhoe_full_minimum, 1e-12
        )
        x = grid.left() + zeff_fraction * grid.width()
        # rho_e increases upward, so fraction 0 is at the bottom of the grid.
        y = grid.top() + (1.0 - rhoe_fraction) * grid.height()
        return x, y

    def _pixel_to_data(self, x, y):
        grid = self._grid_rect()
        zeff_fraction = (x - grid.left()) / max(grid.width(), 1e-6)
        rhoe_fraction = 1.0 - (y - grid.top()) / max(grid.height(), 1e-6)
        zeff_value = self._zeff_full_minimum + zeff_fraction * (
            self._zeff_full_maximum - self._zeff_full_minimum
        )
        rhoe_value = self._rhoe_full_minimum + rhoe_fraction * (
            self._rhoe_full_maximum - self._rhoe_full_minimum
        )
        return zeff_value, rhoe_value

    def _selection_rect_px(self):
        zeff_min, zeff_max, rhoe_min, rhoe_max = self._selection
        x0, y0 = self._data_to_pixel(zeff_min, rhoe_max)
        x1, y1 = self._data_to_pixel(zeff_max, rhoe_min)
        return QtCore.QRectF(
            min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
        )

    def _hit_test(self, pos):
        rect = self._selection_rect_px()
        h = self._HANDLE_PIXELS
        x, y = pos.x(), pos.y()

        near_left = abs(x - rect.left()) <= h
        near_right = abs(x - rect.right()) <= h
        near_top = abs(y - rect.top()) <= h
        near_bottom = abs(y - rect.bottom()) <= h
        inside_x = rect.left() - h <= x <= rect.right() + h
        inside_y = rect.top() - h <= y <= rect.bottom() + h

        if near_left and near_top and inside_x and inside_y:
            return "top_left"
        if near_right and near_top and inside_x and inside_y:
            return "top_right"
        if near_left and near_bottom and inside_x and inside_y:
            return "bottom_left"
        if near_right and near_bottom and inside_x and inside_y:
            return "bottom_right"
        if near_left and inside_y:
            return "left"
        if near_right and inside_y:
            return "right"
        if near_top and inside_x:
            return "top"
        if near_bottom and inside_x:
            return "bottom"
        if rect.contains(pos):
            return "move"
        return "new"

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        self._drag_mode = self._hit_test(pos)
        self._drag_start_pos = pos
        self._drag_start_selection = list(self._selection)

        if self._drag_mode == "new":
            zeff_value, rhoe_value = self._pixel_to_data(pos.x(), pos.y())
            self._selection = [
                zeff_value, zeff_value, rhoe_value, rhoe_value,
            ]
            self._drag_mode = "bottom_right"
            self._drag_start_selection = list(self._selection)
        self.update()

    def mouseMoveEvent(self, event):
        cursor_by_mode = {
            "move": QtCore.Qt.CursorShape.SizeAllCursor,
            "left": QtCore.Qt.CursorShape.SizeHorCursor,
            "right": QtCore.Qt.CursorShape.SizeHorCursor,
            "top": QtCore.Qt.CursorShape.SizeVerCursor,
            "bottom": QtCore.Qt.CursorShape.SizeVerCursor,
            "top_left": QtCore.Qt.CursorShape.SizeFDiagCursor,
            "bottom_right": QtCore.Qt.CursorShape.SizeFDiagCursor,
            "top_right": QtCore.Qt.CursorShape.SizeBDiagCursor,
            "bottom_left": QtCore.Qt.CursorShape.SizeBDiagCursor,
        }

        if self._drag_mode is None:
            hover_mode = self._hit_test(event.position())
            self.setCursor(cursor_by_mode.get(
                hover_mode, QtCore.Qt.CursorShape.CrossCursor
            ))
            return

        pos = event.position()
        dx = pos.x() - self._drag_start_pos.x()
        dy = pos.y() - self._drag_start_pos.y()
        zeff_min0, zeff_max0, rhoe_min0, rhoe_max0 = self._drag_start_selection

        grid = self._grid_rect()
        zeff_span = self._zeff_full_maximum - self._zeff_full_minimum
        rhoe_span = self._rhoe_full_maximum - self._rhoe_full_minimum
        d_zeff = dx / max(grid.width(), 1e-6) * zeff_span
        # Screen y grows downward, rho_e grows upward -> invert the sign.
        d_rhoe = -dy / max(grid.height(), 1e-6) * rhoe_span

        zeff_min, zeff_max = zeff_min0, zeff_max0
        rhoe_min, rhoe_max = rhoe_min0, rhoe_max0

        if self._drag_mode == "move":
            zeff_min, zeff_max = zeff_min0 + d_zeff, zeff_max0 + d_zeff
            rhoe_min, rhoe_max = rhoe_min0 + d_rhoe, rhoe_max0 + d_rhoe
        else:
            if "left" in self._drag_mode:
                zeff_min = zeff_min0 + d_zeff
            if "right" in self._drag_mode:
                zeff_max = zeff_max0 + d_zeff
            if "top" in self._drag_mode:
                rhoe_max = rhoe_max0 + d_rhoe
            if "bottom" in self._drag_mode:
                rhoe_min = rhoe_min0 + d_rhoe

        # Keep the rectangle inside the full stack bounds.
        zeff_min = max(self._zeff_full_minimum, min(zeff_min, self._zeff_full_maximum))
        zeff_max = max(self._zeff_full_minimum, min(zeff_max, self._zeff_full_maximum))
        rhoe_min = max(self._rhoe_full_minimum, min(rhoe_min, self._rhoe_full_maximum))
        rhoe_max = max(self._rhoe_full_minimum, min(rhoe_max, self._rhoe_full_maximum))

        if zeff_min > zeff_max:
            zeff_min, zeff_max = zeff_max, zeff_min
        if rhoe_min > rhoe_max:
            rhoe_min, rhoe_max = rhoe_max, rhoe_min

        self._selection = [zeff_min, zeff_max, rhoe_min, rhoe_max]
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        was_dragging = self._drag_mode is not None
        self._drag_mode = None
        self._drag_start_pos = None
        self._drag_start_selection = None
        if was_dragging:
            self.selectionChanged.emit(*self._selection)

    def resizeEvent(self, event):
        self._background_pixmap = None
        super().resizeEvent(event)

    def paintEvent(self, event):
        self._ensure_background_pixmap()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), self.palette().window())

        grid = self._grid_rect()
        if self._background_pixmap is not None:
            painter.drawPixmap(grid.topLeft(), self._background_pixmap)
        painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black))
        painter.drawRect(grid)

        # Selection rectangle overlay (dashed, to mark it as an
        # adjustable/interactive selection).
        selection_rect = self._selection_rect_px()
        overlay_pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        overlay_pen.setWidth(2)
        overlay_pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(overlay_pen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 40))
        painter.drawRect(selection_rect)

        # Probed points, marked as small black-outlined white dots.
        if self._points:
            dot_pen = QtGui.QPen(QtCore.Qt.GlobalColor.black)
            dot_pen.setWidth(1)
            painter.setPen(dot_pen)
            painter.setBrush(QtGui.QColor(255, 255, 255, 230))
            for zeff_value, rhoe_value in self._points:
                if not (
                    self._zeff_full_minimum <= zeff_value <= self._zeff_full_maximum
                    and self._rhoe_full_minimum <= rhoe_value <= self._rhoe_full_maximum
                ):
                    continue
                px, py = self._data_to_pixel(zeff_value, rhoe_value)
                painter.drawEllipse(QtCore.QPointF(px, py), 4, 4)

        # Axis ticks and numeric labels -- white, so they stay legible
        # against Dragonfly's dark theme (the panel background is dark,
        # black text was effectively invisible there). Font size scales
        # with the widget's own size (baseline: 8pt at 260px, the small
        # in-panel size) so the temporary larger instance built for
        # "Enlarge / Save" comes out with genuinely bigger, readable
        # numbers instead of the same tiny 8pt stretched onto a bigger
        # canvas. Damped (power 0.7, not linear) and capped, and the
        # title is tied to the tick size rather than scaling on its
        # own -- a plain linear scale made the axis titles balloon out
        # far past the plot width at 820px.
        size_scale = self._size_scale()
        tick_font_size = max(8, min(18, round(8 * size_scale ** 0.7)))
        title_font_size = min(tick_font_size + 5, 22)

        font = painter.font()
        font.setPointSize(tick_font_size)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.white))

        tick_count = 4
        tick_label_height = tick_font_size + 10
        for index in range(tick_count):
            fraction = index / (tick_count - 1)

            x = grid.left() + fraction * grid.width()
            zeff_value = self._zeff_full_minimum + fraction * (
                self._zeff_full_maximum - self._zeff_full_minimum
            )
            painter.drawLine(
                int(x), int(grid.bottom()), int(x), int(grid.bottom()) + 5
            )
            painter.drawText(
                QtCore.QRectF(
                    x - tick_font_size * 4, grid.bottom() + 6,
                    tick_font_size * 8, tick_label_height,
                ),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                MainFormGeo_dect._format_tick_value(zeff_value),
            )

            y = grid.top() + grid.height() * (1.0 - fraction)
            rhoe_value = self._rhoe_full_minimum + fraction * (
                self._rhoe_full_maximum - self._rhoe_full_minimum
            )
            painter.drawLine(
                int(grid.left()) - 5, int(y), int(grid.left()), int(y)
            )
            # rhoe_value is already the small display-scaled number (e.g.
            # 0.97, matching what the standalone rho_e panel's Minimum/
            # Maximum boxes show) -- show it plainly here too, the same
            # way those boxes do (a plain number plus the "(x10^23)" unit
            # spelled out once, in the axis title below, not baked into
            # every individual tick). Multiplying by the scale here and
            # reformatting in scientific notation was the actual bug:
            # depending on what the scale held, it produced either the
            # unscaled small number relabelled as if it had its own tiny
            # exponent, or a technically-correct but unfamiliar huge
            # exponent (~10^22) that doesn't match this convention
            # anywhere else in the plugin.
            painter.drawText(
                QtCore.QRectF(
                    0, y - tick_label_height / 2,
                    grid.left() - 8, tick_label_height,
                ),
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter,
                MainFormGeo_dect._format_tick_value(rhoe_value),
            )

        MainFormGeo_dect._draw_html_centered(
            painter,
            "<div style='color:#ffffff; font-size:{0}pt; "
            "font-weight:700; text-align:center;'>"
            "<i>Z<sub>eff</sub></i></div>".format(title_font_size),
            QtCore.QRectF(
                grid.left(), grid.bottom() + tick_label_height + 8,
                grid.width(), title_font_size + 10,
            ),
        )

        painter.save()
        painter.translate(title_font_size + 6, grid.top() + grid.height() / 2)
        painter.rotate(-90)
        # Always show "x10^23", the same way the standalone rho_e panel's
        # Minimum/Maximum boxes always add that suffix unconditionally
        # (see their setSuffix() call) rather than checking the actual
        # scale value -- matching that existing, unconditional convention
        # keeps this label consistent with the rest of the plugin instead
        # of disagreeing with it in edge cases.
        MainFormGeo_dect._draw_html_centered(
            painter,
            "<div style='color:#ffffff; font-size:{0}pt; "
            "font-weight:700; text-align:center;'>"
            "<i>\u03c1<sub>e</sub></i> (\u00d710\u00b2\u00b3 "
            "e\u207b\u00b7cm\u207b\u00b3)</div>".format(title_font_size),
            QtCore.QRectF(
                -grid.height() / 2, -(title_font_size + 10) / 2,
                grid.height(), title_font_size + 10,
            ),
        )
        painter.restore()

        painter.end()


class GeoDECTRedistributionPreview(QtWidgets.QWidget):
    """
    Small live preview showing how the selected sub-range will look once
    "redistributed" across the full colour palette (what Apply will
    actually produce), next to the main scatter selector which always
    shows colours over the FULL absolute stack range. This exists because
    the redistribution can be visually surprising: a sub-range picked from
    a blue/green area of the full-range legend can still show the full
    rainbow (including pink/red) once its own min/max are stretched to
    fill the whole palette again.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zeff_minimum = 0.0
        self._zeff_maximum = 1.0
        self._rhoe_minimum = 0.0
        self._rhoe_maximum = 1.0
        self._pixmap = None
        self.setMinimumSize(140, 140)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setFixedSize(160, 160)

    def set_range(self, zeff_minimum, zeff_maximum, rhoe_minimum, rhoe_maximum):
        self._zeff_minimum = float(zeff_minimum)
        self._zeff_maximum = float(zeff_maximum)
        self._rhoe_minimum = float(rhoe_minimum)
        self._rhoe_maximum = float(rhoe_maximum)
        self._pixmap = None
        self.update()

    def paintEvent(self, event):
        if (
            self._pixmap is None
            and self._zeff_maximum > self._zeff_minimum
            and self._rhoe_maximum > self._rhoe_minimum
        ):
            rgb = create_geodect_legend(
                self._zeff_minimum,
                self._zeff_maximum,
                self._rhoe_minimum,
                self._rhoe_maximum,
                width=self.width(),
                height=self.height(),
            )
            self._pixmap = MainFormGeo_dect._numpy_rgb_to_pixmap(rgb)

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        if self._pixmap is not None:
            painter.drawPixmap(0, 0, self._pixmap)
        border_pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        border_pen.setWidth(2)
        border_pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(border_pen)
        painter.drawRect(1, 1, self.width() - 3, self.height() - 3)
        painter.end()


class ScientificDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """A QDoubleSpinBox that displays very large or very small values in
    scientific notation instead of Qt's default plain-decimal formatting.

    Qt's QDoubleSpinBox has no built-in scientific-notation display: no
    matter how "decimals" is set, it always writes the full number out
    -- for a value like 1.115718e+23 that means a ~26-character wall of
    digits, which is exactly what was making the rho_e range boxes
    illegible. This keeps plain notation for ordinary-sized values (so
    Z_eff, or anything already comfortably scaled, is unaffected) and
    only switches to scientific notation once a value is large/small
    enough that plain decimal would be unreadable either way.
    """

    def textFromValue(self, value):
        if value != 0 and (abs(value) >= 1.0e5 or abs(value) < 1.0e-4):
            return "{:.6e}".format(value)
        return super().textFromValue(value)

    def valueFromText(self, text):
        try:
            return float(text.split(" ")[0].replace(",", ""))
        except (ValueError, IndexError):
            return super().valueFromText(text)

    def validate(self, text, pos):
        return (QtGui.QValidator.State.Acceptable, text, pos)


class MainFormGeo_dect(OrsAbstractWindow):

    def __init__(self, implementation, parent=None):
        super().__init__(implementation, parent)

        self.ui = Ui_MainFormGeo_dect()
        self.ui.setupUi(self)

        # Allow the Geo_DECT panel to expand without forcing the
        # Dragonfly dock to be taller than the available screen space.
        self.setMinimumWidth(300)
        self.setMinimumHeight(0)
        self.setMaximumSize(16777215, 16777215)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        self.low_format = None
        self.high_format = None
        self.low_calibration = None
        self.high_calibration = None
        self.low_files = []
        self.high_files = []
        self._full_preview_pixmap = None
        self._preview_dialog = None
        self._visualization_dialog = None
        self.geodect_3d_channels = {}
        self.geodect_3d_visuals = {}
        self.geodect_3d_clip_boxes = {}
        self.geodect_3d_legends = {}
        self.geodect_3d_scalar_collections = {}
        self.geodect_3d_luts = {}
        self.geodect_3d_display_settings = {}
        self.geodect_3d_scene_layout = None
        self.geodect_3d_current_view = None
        # Compact Dragonfly-style dock for scalar display ranges (slider +
        # Apply, same pattern as the "combined" property).
        self._display_range_dock = None
        self._display_range_controls = {}

        self.calibration_profiles_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "geodect_stoichiometric_calibrations.json",
        )
        self.calibration_profiles = {}
        self.last_calibration_profile = ""

        self._build_interface()
        self._connect_signals()
        self._load_calibration_profiles_from_disk()
        self._refresh_calibration_profile_combo()
        self._load_last_calibration_profile()

        WorkingContext.registerOrsWidget(
            "Geo_DECT_7e90f2c77eb011f18afab07b2518f399",
            implementation,
            "MainFormGeo_dect",
            self
        )

        # Deferred (not called directly here): at this exact point in
        # __init__, Dragonfly may not have finished docking this widget
        # into its main window yet, and _show_live_display_range_dock()
        # relies on self.window() already resolving to that main window
        # to attach the right-hand panel in the right place. Running
        # this on the next event loop cycle instead of synchronously
        # gives Dragonfly time to finish that first.
        QtCore.QTimer.singleShot(
            0, self._reconnect_existing_geodect_channels
        )

    def _reconnect_existing_geodect_channels(self, force_recreate=False):
        """Re-attach colouring and the display-range panel to Geo_DECT
        channels that already exist when the plugin opens -- typically
        because a Dragonfly project containing them was just reopened.

        The channel DATA survives a project save (it's a native
        Channel); the VisualChannel, the LUT and the right-hand dock do
        not -- they are created fresh by Build()/Apply() and only ever
        live in this plugin's own Python-side state. Without this, a
        reopened project shows the Geo_DECT volumes grey with no panel.

        Best-effort only: any failure here (including simply finding no
        matching channels, the normal case for a fresh session) never
        blocks the plugin from opening -- but unlike a silent no-op,
        every early exit and every exception now leaves a trace in
        volume_status_label (and, on exception, a diagnostic file next
        to this script) so a failed reconnect is visible instead of
        looking identical to "nothing to reconnect".
        """
        diagnostic_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "geodect_reconnect_diagnostic.txt",
        )

        def report(message):
            try:
                self.volume_status_label.setText(
                    "Geo_DECT — reconnect: {}".format(message)
                )
            except Exception:
                pass

        try:
            from . import geodect_3d as geodect3d

            search_diagnostic_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "geodect_channel_search_diagnostic.txt",
            )
            found, found_visuals = geodect3d.find_existing_geodect_channels(
                diagnostic_path=search_diagnostic_path
            )
            missing = [
                key for key in ("zeff", "rhoe", "combined")
                if key not in found
            ]
            if missing:
                if found:
                    report(
                        "found {} but not {} -- not reconnecting "
                        "(check channel titles in the Properties "
                        "panel).".format(
                            sorted(found.keys()), missing
                        )
                    )
                else:
                    report(
                        "no existing Geo_DECT channels found in this "
                        "scene. Details (all channel titles Dragonfly "
                        "reported) saved to "
                        "geodect_channel_search_diagnostic.txt next to "
                        "the plugin scripts."
                    )
                return

            from OrsHelpers.volumehelper import VolumeHelper
            from OrsHelpers.layoutHelper import LayoutHelper
            from OrsLibraries.workingcontext import WorkingContext
            from OrsHelpers.layoutpropertieshelper import (
                LayoutPropertiesHelper,
            )
            from ORSServiceClass.preferencesapplier.preferencesApplyManager import (
                PreferencesApplyManager,
            )

            current_view = WorkingContext.getCurrentView(None)
            if current_view is None:
                report(
                    "found the 3 channels, but Dragonfly has no "
                    "current view yet -- click once in a viewport "
                    "of this project, then reopen Geo_DECT."
                )
                return
            scene_layout = LayoutHelper.getSceneLayoutOfView(current_view)
            if scene_layout is None:
                report(
                    "found the 3 channels and a current view, but "
                    "could not resolve its scene layout."
                )
                return

            layout_diagnostic_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "geodect_layout_search_diagnostic.txt",
            )
            layout_visuals = geodect3d.find_visual_channels_in_layout(
                scene_layout, diagnostic_path=layout_diagnostic_path
            )
            # Prefer whatever was found nested inside scene_layout
            # itself over the global sweep -- see
            # find_visual_channels_in_layout()'s docstring.
            for key, node in layout_visuals.items():
                found_visuals[key] = node

            visuals = {}
            channel_diag_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "geodect_reconnect_channel_diagnostic.txt",
            )
            try:
                with open(channel_diag_path, "w", encoding="utf-8"):
                    pass
            except Exception:
                pass
            already_had = (
                {}
                if force_recreate
                else (getattr(self, "geodect_3d_visuals", {}) or {})
            )
            for key in ("rhoe", "combined", "zeff"):
                # This-session reuse: if a previous call (automatic on
                # open, or an earlier press of this same button) already
                # created a VisualChannel for this key, reuse THAT one
                # instead of creating yet another one -- otherwise every
                # extra click piles up more duplicate VisualChannels on
                # the same Channel.
                own_existing_visual = already_had.get(key)
                existing_visual = (
                    own_existing_visual
                    if own_existing_visual is not None
                    else found_visuals.get(key)
                )
                if existing_visual is not None:
                    # Confirmed 2026-09: the VisualChannel created by a
                    # previous Build survives a Dragonfly project save/
                    # reload -- reuse it instead of creating a second,
                    # separate one. A second VisualChannel would be the
                    # one this plugin colors, while the Properties
                    # panel keeps showing/toggling the original (stuck
                    # grey), so nothing the person does would appear to
                    # have any visible effect.
                    visuals[key] = existing_visual
                    continue

                try:
                    channel_diag_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "geodect_reconnect_channel_diagnostic.txt",
                    )
                    with open(
                        channel_diag_path, "a", encoding="utf-8"
                    ) as handle:
                        ch = found[key]
                        handle.write(
                            "{}: isDataInitialized={}, "
                            "isDataTypeFloat={}, "
                            "isDataTypeUnsignedShort={}, "
                            "size=({},{},{}), "
                            "spacing=({},{},{}), "
                            "minmax=({},{})\n".format(
                                key,
                                ch.getIsDataInitialized(),
                                ch.getIsDataTypeFloat(),
                                ch.getIsDataTypeUnsignedShort(),
                                ch.getXSize(), ch.getYSize(),
                                ch.getZSize(),
                                ch.getXSpacing(), ch.getYSpacing(),
                                ch.getZSpacing(),
                                ch.getMinimumValue(),
                                ch.getMaximumValue(),
                            )
                        )
                except Exception as exc:
                    try:
                        with open(
                            channel_diag_path, "a", encoding="utf-8"
                        ) as handle:
                            handle.write(
                                "{}: property dump FAILED: {}: {}\n"
                                .format(key, type(exc).__name__, exc)
                            )
                    except Exception:
                        pass

                # Confirmed 2026-09: a Channel reloaded from a saved
                # Dragonfly project (as opposed to one created fresh in
                # this same session via Channel().fromNDArray(), which
                # is what Build always does) never builds a 3D texture,
                # even though every property checked above looks
                # identical and no step ever raises an exception. A
                # normal Build in the same session, on the same kind of
                # data, works every time. There is no known way to fix
                # the reloaded Channel object itself, so when the
                # person explicitly asks to reconnect (force_recreate),
                # its data is instead copied into a brand new Channel
                # -- the exact same function Build itself uses -- and
                # THAT is what gets built into a volume from here on.
                # This does leave the old (still grey, still
                # unusable-in-3D) channel behind under the same title;
                # the person removes it manually from the Properties
                # panel once the new one is confirmed working.
                build_channel = found[key]
                if force_recreate:
                    try:
                        source_channel = found[key]
                        source_array = np.asarray(
                            source_channel.getNDArray()
                        )
                        source_dtype = (
                            np.float32
                            if source_channel.getIsDataTypeFloat()
                            else np.uint16
                        )
                        spacing_xyz_m = (
                            float(source_channel.getXSpacing()),
                            float(source_channel.getYSpacing()),
                            float(source_channel.getZSpacing()),
                        )
                        build_channel = geodect3d.create_native_channel(
                            source_array,
                            source_dtype,
                            source_channel.getTitle(),
                            spacing_xyz_m,
                        )
                        # The new Channel starts with no tags at all --
                        # without copying these over, the chosen range
                        # and discrete/continuous state recorded on the
                        # OLD channel would be silently lost, and the
                        # data (a straight copy, whatever state it was
                        # actually in) would be misread as untouched
                        # physical values even when it is really
                        # already-baked display codes.
                        old_settings, old_state = (
                            geodect3d.read_persisted_display_settings(
                                source_channel
                            )
                        )
                        geodect3d.persist_display_settings(
                            build_channel,
                            old_settings if old_settings else {},
                            old_state,
                        )
                        found[key] = build_channel
                        with open(
                            channel_diag_path, "a", encoding="utf-8"
                        ) as handle:
                            handle.write(
                                "{}: recreated fresh Channel from "
                                "reloaded data -> OK\n".format(key)
                            )
                    except Exception as exc:
                        report(
                            "recreating the '{}' channel failed: "
                            "{}: {}".format(
                                key, type(exc).__name__, exc
                            )
                        )
                        return

                visual, _clip_box, _legend = (
                    VolumeHelper.createVolumeForChannel(
                        build_channel, layout=scene_layout,
                    )
                )
                if visual is None:
                    report(
                        "VolumeHelper.createVolumeForChannel returned "
                        "no VisualChannel for '{}'.".format(key)
                    )
                    return
                visuals[key] = visual

                init_visual = getattr(visual, "initializeVisual", None)
                if callable(init_visual):
                    try:
                        init_visual()
                        with open(
                            channel_diag_path, "a", encoding="utf-8"
                        ) as handle:
                            handle.write(
                                "{}: visual.initializeVisual() -> "
                                "OK\n".format(key)
                            )
                    except Exception as exc:
                        try:
                            with open(
                                channel_diag_path, "a", encoding="utf-8"
                            ) as handle:
                                handle.write(
                                    "{}: visual.initializeVisual() "
                                    "FAILED: {}: {}\n".format(
                                        key, type(exc).__name__, exc
                                    )
                                )
                        except Exception:
                            pass

                try:
                    PreferencesApplyManager.applyPreferencesOn(visual)
                except Exception:
                    # Cosmetic only -- must not discard an otherwise
                    # valid visual representation.
                    pass

                # Confirmed in Build's own code: called DIRECTLY on the
                # VisualChannel (not via LayoutPropertiesHelper, which
                # is what the later LUT-application loop below uses)
                # immediately after creation. This looks like it seeds
                # the volume's own window/level bounds -- separate from
                # colouring -- and Build never skips it. The exact
                # values here don't matter much (the LUT-application
                # loop below sets the real ones per channel state); the
                # native range is used just to have SOME valid bounds
                # from the very start, matching what Build always has
                # by this point.
                try:
                    native_min = float(build_channel.getMinimumValue())
                    native_max = float(build_channel.getMaximumValue())
                    native_width = native_max - native_min
                    native_center = (native_max + native_min) / 2.0
                    visual.set2DWindowLevelValuesForAllViews(
                        native_width, native_center
                    )
                    visual.set3DWindowLevelValuesForAllViews(
                        native_width, native_center
                    )
                except Exception:
                    pass

                # Only a freshly created VisualChannel is forced hidden
                # here (matching Build's own default-hidden behaviour
                # for a brand new volume) -- an EXISTING, reused one
                # keeps whatever visibility the person already had.
                for display in scene_layout.getAllChildViews():
                    visual.setIsVisibleForView(display, False)

                # Without this, Dragonfly never builds the 3D texture
                # for the volume -- same publish/refresh sequence Build
                # uses right after creating each VisualChannel.
                visual.triggerClassEvent("OrsObjectPublishedEvent")
                visual.setPropertyDirty()
                visual.refreshAllParentViews()

            self.geodect_3d_channels = found
            self.geodect_3d_visuals = visuals
            self.geodect_3d_scene_layout = scene_layout
            self.geodect_3d_current_view = current_view

            try:
                LayoutPropertiesHelper.updateSceneProperties(
                    scene_layout, datasetIsVisible=True,
                )
            except Exception:
                pass

            # _show_live_display_range_dock() needs the FULL native
            # range of each channel as the slider bounds, regardless of
            # discrete/continuous state -- this was missing entirely
            # before, causing a KeyError there even after the LUTs
            # above were applied successfully.
            self.geodect_3d_display_settings["zeff_full_minimum"] = float(
                found["zeff"].getMinimumValue()
            )
            self.geodect_3d_display_settings["zeff_full_maximum"] = float(
                found["zeff"].getMaximumValue()
            )
            self.geodect_3d_display_settings["rhoe_full_minimum"] = float(
                found["rhoe"].getMinimumValue()
            )
            self.geodect_3d_display_settings["rhoe_full_maximum"] = float(
                found["rhoe"].getMaximumValue()
            )

            zeff_palette = np.asarray(
                ZEFF_PALETTE_RGB255, dtype=np.float64
            ) / 255.0
            rhoe_palette = np.asarray(
                RHOE_PALETTE_RGB255, dtype=np.float64
            ) / 255.0
            table_size = (
                geodect3d.SCALAR_FIRST_COLOR_CODE + geodect3d.SCALAR_BINS
            )
            discrete_width = float(table_size - 1)
            discrete_center = discrete_width / 2.0

            luts = {}
            self.geodect_3d_raw_arrays = {}
            raw_arrays_recovered = []
            raw_arrays_lost = []
            for key, palette in (
                ("zeff", zeff_palette), ("rhoe", rhoe_palette),
            ):
                channel = found[key]
                settings, state = geodect3d.read_persisted_display_settings(
                    channel
                )

                # "Apply" on the range panel overwrites the channel's OWN
                # data with integer display codes (see the comment where
                # geodect_3d_raw_arrays is first populated in Build). If
                # that never happened for this channel (still
                # "continuous"), its data is still the true physical
                # values, identical to what raw_arrays held originally --
                # so it can be recovered here, restoring the ability to
                # pick a different range with "Apply" after a reconnect.
                # If it's already "discrete", the physical values are
                # genuinely gone; there is nothing to recover.
                if state == "discrete":
                    raw_arrays_lost.append(key)
                else:
                    self.geodect_3d_raw_arrays[key] = np.asarray(
                        channel.getNDArray()
                    ).astype(np.float32, copy=True)
                    raw_arrays_recovered.append(key)

                if state == "discrete":
                    # The chosen range was already baked into the data
                    # via Apply -- the discrete LUT does not depend on
                    # the range at all, but the dock still needs the
                    # min/max that were selected (persisted alongside
                    # the "discrete" tag itself) to show the sliders in
                    # the right position.
                    lut = geodect3d.create_scalar_context_lut(
                        "Geo_DECT {} display (reconnected)".format(key),
                        palette,
                    )
                    width, center = discrete_width, discrete_center
                    if settings and "{}_minimum".format(key) in settings:
                        self.geodect_3d_display_settings[
                            "{}_minimum".format(key)
                        ] = float(settings["{}_minimum".format(key)])
                        self.geodect_3d_display_settings[
                            "{}_maximum".format(key)
                        ] = float(settings["{}_maximum".format(key)])
                    else:
                        # No persisted range (project saved before this
                        # fix existed) -- fall back to the full native
                        # range so the dock still opens.
                        self.geodect_3d_display_settings[
                            "{}_minimum".format(key)
                        ] = float(channel.getMinimumValue())
                        self.geodect_3d_display_settings[
                            "{}_maximum".format(key)
                        ] = float(channel.getMaximumValue())
                else:
                    if settings:
                        data_min = float(
                            settings["{}_minimum".format(key)]
                        )
                        data_max = float(
                            settings["{}_maximum".format(key)]
                        )
                    else:
                        # Project saved before this fix existed -- no
                        # tag available, fall back to the channel's own
                        # native range rather than leaving it grey.
                        data_min = float(channel.getMinimumValue())
                        data_max = float(channel.getMaximumValue())
                    lut = geodect3d.create_context_scalar_lut(
                        "Geo_DECT {} preview-matched (reconnected)"
                        .format(key),
                        float(channel.getMinimumValue()),
                        float(channel.getMaximumValue()),
                        data_min,
                        data_max,
                        palette,
                    )
                    width = data_max - data_min
                    center = (data_max + data_min) / 2.0
                    self.geodect_3d_display_settings[
                        "{}_minimum".format(key)
                    ] = data_min
                    self.geodect_3d_display_settings[
                        "{}_maximum".format(key)
                    ] = data_max

                LayoutPropertiesHelper.setWindowLevel2D(
                    scene_layout, visuals[key], width, center,
                    logging=False,
                )
                geodect3d.apply_lut_to_channel_in_layout(
                    channel=channel,
                    lut=lut,
                    scene_layout=scene_layout,
                    visual_channel=visuals[key],
                    view=current_view,
                )
                LayoutPropertiesHelper.setWindowLevel3D(
                    scene_layout, visuals[key], width, center,
                    logging=False,
                )
                # NOTE: the 3D LUT is deliberately NOT applied here.
                # apply_confirmed_3d_lut() must run AFTER every other
                # refresh/window-level call below, or Dragonfly silently
                # reverts the 3D coloring without any error (confirmed
                # in _apply_confirmed_3d_coloring, same issue Build
                # avoids with its own 3-second delay) -- see the
                # deferred block further down.
                luts[key] = lut
                visuals[key].setPropertyDirty()
                visuals[key].refreshAllParentViews()

            combined_lut = geodect3d.create_geodect_lut(
                name="Geo_DECT combined (reconnected)",
            )
            geodect3d.apply_confirmed_2d_lut(
                scene_layout, visuals["combined"], combined_lut
            )
            luts["combined"] = combined_lut
            self.geodect_3d_luts = luts
            visuals["combined"].setPropertyDirty()
            visuals["combined"].refreshAllParentViews()

            try:
                current_view.refresh()
            except Exception:
                pass

            # Same staggered-refresh safety net Build uses: Dragonfly
            # needs a moment after volume/LUT creation before a refresh
            # reliably reaches the 3D renderer, so this repeats it a
            # few times instead of relying on the one done above.
            def _refresh_reconnected_visuals():
                try:
                    for a_visual in visuals.values():
                        a_visual.setPropertyDirty()
                        a_visual.refreshAllParentViewsIfVisible()
                        a_visual.refreshAllParentViews()
                    LayoutHelper.refreshAllViewsInSceneLayoutOfView(
                        current_view
                    )
                except Exception:
                    pass

            _refresh_reconnected_visuals()
            QtCore.QTimer.singleShot(0, _refresh_reconnected_visuals)
            QtCore.QTimer.singleShot(120, _refresh_reconnected_visuals)
            QtCore.QTimer.singleShot(350, _refresh_reconnected_visuals)

            # Deferred, and deliberately LAST: apply the 3D LUT only
            # after every refresh above has already run, exactly like
            # Build's own QTimer.singleShot(3000, _apply_confirmed_3d_coloring).
            # Applying it earlier is the reason the 3D volumes stayed
            # uncolored/invisible after reconnecting, even though no
            # exception was ever raised anywhere.
            def _apply_reconnected_3d_coloring():
                diag_lines = []

                def dlog(message):
                    diag_lines.append(str(message))

                try:
                    from OrsHelpers.viewLayoutHelper import (
                        DisplayLayoutHelper,
                    )
                    all_3d_views = (
                        DisplayLayoutHelper.getAll3DViewsInLayout(
                            scene_layout
                        )
                    )
                    dlog(
                        "3D views found in scene_layout: {}".format(
                            len(all_3d_views)
                        )
                    )
                except Exception as exc:
                    dlog(
                        "getAll3DViewsInLayout FAILED: {}: {}".format(
                            type(exc).__name__, exc
                        )
                    )

                for key in ("zeff", "rhoe", "combined"):
                    try:
                        is_visible = []
                        for display in scene_layout.getAllChildViews():
                            try:
                                is_visible.append(
                                    bool(
                                        visuals[key]
                                        .getIsVisibleForView(display)
                                    )
                                )
                            except Exception:
                                pass
                        dlog(
                            "{}: visibility per view before 3D LUT: "
                            "{}".format(key, is_visible)
                        )
                        geodect3d.apply_confirmed_3d_lut(
                            scene_layout, visuals[key], luts[key]
                        )
                        dlog("{}: apply_confirmed_3d_lut -> OK".format(key))
                    except Exception as exc:
                        dlog(
                            "{}: apply_confirmed_3d_lut FAILED: {}: "
                            "{}".format(key, type(exc).__name__, exc)
                        )
                        dlog(traceback.format_exc())
                try:
                    LayoutHelper.refreshAllViewsInSceneLayoutOfView(
                        current_view
                    )
                    dlog("refreshAllViewsInSceneLayoutOfView -> OK")
                except Exception as exc:
                    dlog(
                        "refreshAllViewsInSceneLayoutOfView FAILED: "
                        "{}: {}".format(type(exc).__name__, exc)
                    )

                try:
                    coloring_diagnostic_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "geodect_reconnect_3d_diagnostic.txt",
                    )
                    with open(
                        coloring_diagnostic_path, "w", encoding="utf-8"
                    ) as handle:
                        handle.write("\n".join(diag_lines))
                except Exception:
                    pass

            QtCore.QTimer.singleShot(3000, _apply_reconnected_3d_coloring)

            self._show_live_display_range_dock()
            status_text = (
                "Geo_DECT — reconnected to Z_eff, rho_e and combined "
                "volumes already present in this project "
                "({} VisualChannel(s) reused, {} newly created).".format(
                    len(found_visuals),
                    len(visuals) - len(found_visuals),
                )
            )
            if force_recreate:
                status_text += (
                    " New Z_eff/rho_e/combined channels were created "
                    "for the 3D volume (reloaded channels never "
                    "render in 3D, only freshly created ones do) -- "
                    "once you've confirmed these look right, delete "
                    "the old grey ones with the same names from the "
                    "Properties panel to avoid duplicates."
                )
            if raw_arrays_lost:
                status_text += (
                    " Note: {} already had a chosen range baked in "
                    "before this project was last saved, so its range "
                    "can no longer be changed with Apply in this "
                    "session (rebuild from the original scans for "
                    "that) -- colors and the 3D volume are otherwise "
                    "unaffected.".format(" and ".join(sorted(
                        raw_arrays_lost
                    )))
                )
            self.volume_status_label.setText(status_text)
        except Exception as exc:
            try:
                with open(
                    diagnostic_path, "w", encoding="utf-8"
                ) as handle:
                    handle.write(
                        "Geo_DECT reconnect failed: {}: {}\n\n"
                        .format(type(exc).__name__, exc)
                    )
                    handle.write(traceback.format_exc())
            except Exception:
                pass
            report(
                "failed ({}: {}). Details saved to "
                "geodect_reconnect_diagnostic.txt next to the plugin "
                "scripts.".format(type(exc).__name__, exc)
            )

    def _build_interface(self):
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # Dragonfly controls the width of the docked left panel.
        # Keep a horizontal scrollbar available when the panel is narrower
        # than the Geo_DECT controls.
        self.scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        # Keep the vertical scrollbar visible in the docked panel.
        # This prevents lower controls from being clipped by Dragonfly.
        self.scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.scroll_area.setMinimumHeight(0)
        self.scroll_area.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        self.scroll_content = QtWidgets.QWidget()
        self.scroll_content.setMinimumWidth(520)
        self.scroll_content.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred
        )
        self.main_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        self.main_layout.setContentsMargins(4, 4, 4, 4)
        self.main_layout.setSpacing(8)
        self.main_layout.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetMinimumSize
        )

        self.scroll_area.setWidget(self.scroll_content)
        self.ui.verticalLayout.addWidget(self.scroll_area)

        title = QtWidgets.QLabel("Geo_DECT")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.main_layout.addWidget(title)

        dataset_group = QtWidgets.QGroupBox("Dataset input")
        dataset_layout = QtWidgets.QGridLayout(dataset_group)

        dataset_layout.addWidget(QtWidgets.QLabel("LOW folder"), 0, 0)
        self.low_folder_edit = QtWidgets.QLineEdit()
        self.low_folder_edit.setPlaceholderText("Select LOW folder")
        dataset_layout.addWidget(self.low_folder_edit, 0, 1)
        self.low_browse_button = QtWidgets.QPushButton("Browse...")
        dataset_layout.addWidget(self.low_browse_button, 0, 2)

        dataset_layout.addWidget(QtWidgets.QLabel("HIGH folder"), 1, 0)
        self.high_folder_edit = QtWidgets.QLineEdit()
        self.high_folder_edit.setPlaceholderText("Select HIGH folder")
        dataset_layout.addWidget(self.high_folder_edit, 1, 1)
        self.high_browse_button = QtWidgets.QPushButton("Browse...")
        dataset_layout.addWidget(self.high_browse_button, 1, 2)

        self.check_button = QtWidgets.QPushButton(
            "Check datasets and calibration"
        )
        dataset_layout.addWidget(
            self.check_button, 2, 0, 1, 3
        )

        dataset_layout.addWidget(
            QtWidgets.QLabel("Voxel size override (µm)"), 3, 0
        )
        self.voxel_size_override_edit = QtWidgets.QLineEdit()
        self.voxel_size_override_edit.setPlaceholderText(
            "X, Y, Z (µm) -- auto-detected from metadata, edit if wrong"
        )
        self.voxel_size_override_edit.setToolTip(
            "Voxel size in micrometres. Enter either ONE value "
            "(isotropic voxel, applied to X, Y and Z alike -- typical "
            "for micro-CT) or THREE values 'X, Y, Z' (anisotropic voxel "
            "-- typical for medical DICOM, where the slice spacing Z "
            "usually differs from the in-plane pixel spacing). "
            "Auto-filled from the dataset's own metadata after "
            "'Check datasets and calibration' -- correct it here if that "
            "value looks wrong (e.g. metadata missing or misread), and "
            "it always takes priority over auto-detection at Build time."
        )
        dataset_layout.addWidget(self.voxel_size_override_edit, 3, 1, 1, 2)

        self.reset_button = QtWidgets.QPushButton(
            "Reset current analysis"
        )
        dataset_layout.addWidget(
            self.reset_button, 4, 0, 1, 3
        )

        self.main_layout.addWidget(dataset_group)

        self.manual_calibration_group = QtWidgets.QGroupBox(
            "Manual physical intensity calibration"
        )
        calibration_layout = QtWidgets.QGridLayout(self.manual_calibration_group)

        calibration_layout.addWidget(QtWidgets.QLabel(""), 0, 0)
        calibration_layout.addWidget(QtWidgets.QLabel("Physical min"), 0, 1)
        calibration_layout.addWidget(QtWidgets.QLabel("Physical max"), 0, 2)

        calibration_layout.addWidget(
            self._html_label("<i>&mu;</i><sub><i>LOW</i></sub>"), 1, 0
        )
        self.low_min_edit = QtWidgets.QLineEdit()
        self.low_max_edit = QtWidgets.QLineEdit()
        self.low_min_edit.setPlaceholderText("e.g. -1.75586")
        self.low_max_edit.setPlaceholderText("e.g. 6.24414")
        calibration_layout.addWidget(self.low_min_edit, 1, 1)
        calibration_layout.addWidget(self.low_max_edit, 1, 2)

        calibration_layout.addWidget(
            self._html_label("<i>&mu;</i><sub><i>HIGH</i></sub>"), 2, 0
        )
        self.high_min_edit = QtWidgets.QLineEdit()
        self.high_max_edit = QtWidgets.QLineEdit()
        self.high_min_edit.setPlaceholderText("e.g. -2.26953")
        self.high_max_edit.setPlaceholderText("e.g. 5.73047")
        calibration_layout.addWidget(self.high_min_edit, 2, 1)
        calibration_layout.addWidget(self.high_max_edit, 2, 2)

        for widget in (
            self.low_min_edit,
            self.low_max_edit,
            self.high_min_edit,
            self.high_max_edit,
        ):
            widget.setEnabled(False)

        self.main_layout.addWidget(self.manual_calibration_group)

        stoich_group = QtWidgets.QGroupBox("Stoichiometric calibration")
        stoich_layout = QtWidgets.QVBoxLayout(stoich_group)

        degree_layout = QtWidgets.QGridLayout()
        degree_layout.addWidget(QtWidgets.QLabel("Polynomial degree"), 0, 0)
        degree_layout.addWidget(
            self._html_label("<i>Z</i><sub><i>eff</i></sub>"), 0, 1
        )
        degree_layout.addWidget(
            self._html_label("<i>&rho;</i><sub><i>e</i></sub>"), 0, 2
        )

        self.zeff_degree_combo = QtWidgets.QComboBox()
        self.rhoe_degree_combo = QtWidgets.QComboBox()
        for degree in range(6):
            self.zeff_degree_combo.addItem(str(degree))
            self.rhoe_degree_combo.addItem(str(degree))
        self.zeff_degree_combo.setCurrentIndex(2)
        self.rhoe_degree_combo.setCurrentIndex(2)
        degree_layout.addWidget(self.zeff_degree_combo, 1, 1)
        degree_layout.addWidget(self.rhoe_degree_combo, 1, 2)
        stoich_layout.addLayout(degree_layout)

        coeff_layout = QtWidgets.QGridLayout()
        coeff_layout.addWidget(QtWidgets.QLabel("Power"), 0, 0)
        coeff_layout.addWidget(
            self._html_label("<i>Z</i><sub><i>eff</i></sub> coefficient"), 0, 1
        )
        coeff_layout.addWidget(
            self._html_label("<i>&rho;</i><sub><i>e</i></sub> coefficient"), 0, 2
        )

        self.zeff_coeff_edits = []
        self.rhoe_coeff_edits = []
        for degree in range(6):
            power_label = QtWidgets.QLabel(str(degree))
            power_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            zeff_edit = QtWidgets.QLineEdit()
            rhoe_edit = QtWidgets.QLineEdit()
            zeff_edit.setPlaceholderText(f"a{degree}")
            rhoe_edit.setPlaceholderText(f"d{degree}")
            coeff_layout.addWidget(power_label, degree + 1, 0)
            coeff_layout.addWidget(zeff_edit, degree + 1, 1)
            coeff_layout.addWidget(rhoe_edit, degree + 1, 2)
            self.zeff_coeff_edits.append(zeff_edit)
            self.rhoe_coeff_edits.append(rhoe_edit)
        stoich_layout.addLayout(coeff_layout)

        ratio_group = QtWidgets.QGroupBox(
            "\u03bcLOW/\u03bcHIGH plausible range"
        )
        ratio_layout = QtWidgets.QGridLayout(ratio_group)
        ratio_layout.addWidget(QtWidgets.QLabel("Minimum"), 0, 0)
        ratio_layout.addWidget(QtWidgets.QLabel("Maximum"), 0, 1)
        self.ratio_min_edit = QtWidgets.QLineEdit("0.9")
        self.ratio_max_edit = QtWidgets.QLineEdit("1.4")
        self.ratio_min_edit.setToolTip(
            "Voxels whose mu_LOW/mu_HIGH ratio falls outside "
            "[minimum, maximum] are excluded before the calibration "
            "polynomial is evaluated, instead of only checking the "
            "resulting Zeff/rho_e afterwards. An extreme ratio caused "
            "by a partial-volume (mixel) voxel or ordinary noise can "
            "otherwise land near the curve's vertex and be mistaken "
            "for a plausible result. Set to match the range actually "
            "spanned by your calibration standards, with a margin. "
            "Leave empty to disable this check."
        )
        self.ratio_max_edit.setToolTip(self.ratio_min_edit.toolTip())
        ratio_layout.addWidget(self.ratio_min_edit, 1, 0)
        ratio_layout.addWidget(self.ratio_max_edit, 1, 1)
        stoich_layout.addWidget(ratio_group)

        denoise_group = QtWidgets.QGroupBox(
            "Noise reduction before the \u03bcLOW/\u03bcHIGH ratio"
        )
        denoise_layout = QtWidgets.QGridLayout(denoise_group)
        denoise_layout.addWidget(QtWidgets.QLabel("Filter"), 0, 0)
        self.denoise_method_combo = QtWidgets.QComboBox()
        self.denoise_method_combo.addItem("None (raw voxels)", "none")
        self.denoise_method_combo.addItem(
            "Gaussian, mask-aware (recommended)", "gaussian"
        )
        self.denoise_method_combo.addItem("Median", "median")
        denoise_layout.addWidget(self.denoise_method_combo, 0, 1)
        self.denoise_strength_label = QtWidgets.QLabel("Sigma (px)")
        denoise_layout.addWidget(self.denoise_strength_label, 1, 0)
        self.denoise_strength_spin = QtWidgets.QDoubleSpinBox()
        self.denoise_strength_spin.setRange(0.5, 10.0)
        self.denoise_strength_spin.setSingleStep(0.5)
        self.denoise_strength_spin.setDecimals(1)
        self.denoise_strength_spin.setValue(1.5)
        denoise_layout.addWidget(self.denoise_strength_spin, 1, 1)
        denoise_tooltip = (
            "The same filter is applied to mu_LOW and mu_HIGH, slice by "
            "slice, before their ratio is formed -- in the single-slice "
            "test, in the 3D Build and therefore in the exported "
            "statistics. The ratio amplifies the noise of both images "
            "and the Zeff polynomial amplifies it again: without "
            "filtering, per-voxel scatter can merge distinct mineral "
            "phases into one broad histogram peak. Gaussian: sigma in "
            "pixels (1-2 is usually enough). Median: window size in "
            "pixels (odd, >= 3). Only retained voxels are used, so the "
            "background never bleeds into the sample edge."
        )
        self.denoise_method_combo.setToolTip(denoise_tooltip)
        self.denoise_strength_spin.setToolTip(denoise_tooltip)

        def _update_denoise_controls(_index=None):
            method = self.denoise_method_combo.currentData()
            self.denoise_strength_spin.setEnabled(method != "none")
            if method == "median":
                self.denoise_strength_label.setText("Window (px)")
                self.denoise_strength_spin.setDecimals(0)
                self.denoise_strength_spin.setSingleStep(2.0)
                if self.denoise_strength_spin.value() < 3:
                    self.denoise_strength_spin.setValue(3)
            else:
                self.denoise_strength_label.setText("Sigma (px)")
                self.denoise_strength_spin.setDecimals(1)
                self.denoise_strength_spin.setSingleStep(0.5)

        self.denoise_method_combo.currentIndexChanged.connect(
            _update_denoise_controls
        )
        _update_denoise_controls()
        stoich_layout.addWidget(denoise_group)

        numerator_group = QtWidgets.QGroupBox(
            "Attenuation coefficient used in the numerator of ρe"
        )
        numerator_layout = QtWidgets.QVBoxLayout(numerator_group)
        self.rhoe_mu_low_radio = QtWidgets.QRadioButton("Use μLOW")
        self.rhoe_mu_high_radio = QtWidgets.QRadioButton("Use μHIGH")
        self.rhoe_mu_mean_radio = QtWidgets.QRadioButton(
            "Use the mean of μLOW and μHIGH"
        )
        self.rhoe_mu_low_radio.setChecked(True)
        numerator_layout.addWidget(self.rhoe_mu_low_radio)
        numerator_layout.addWidget(self.rhoe_mu_high_radio)
        numerator_layout.addWidget(self.rhoe_mu_mean_radio)
        self.rhoe_relative_to_water_check = QtWidgets.QCheckBox(
            "Convert \u03c1e / \u03c1e,water to absolute "
            "e\u207b\u00b7cm\u207b\u00b3 (only if the calibration "
            "returns relative values \u2014 detected automatically)"
        )
        self.rhoe_relative_to_water_check.setChecked(True)
        self.rhoe_relative_to_water_check.setToolTip(
            "Equation (6) of the stoichiometric method normalises rho_e by "
            "the electron density of water, so the calibration returns "
            "rho_e / rho_e,water (values of a few units). When checked "
            "and the computed values are of that order, they are "
            "multiplied by rho_e,water = 3.343e23 e-/cm3 and shown as "
            "x10^23 e-.cm-3 everywhere. Profiles that already return "
            "absolute rho_e (values ~1e23) are detected and left "
            "unchanged, so nothing is ever converted twice."
        )
        numerator_layout.addWidget(self.rhoe_relative_to_water_check)
        stoich_layout.addWidget(numerator_group)

        profile_group = QtWidgets.QGroupBox(
            "Saved stoichiometric calibration profiles"
        )
        profile_layout = QtWidgets.QGridLayout(profile_group)

        profile_layout.addWidget(
            QtWidgets.QLabel("Saved profile"),
            0,
            0,
        )
        self.calibration_profile_combo = QtWidgets.QComboBox()
        self.calibration_profile_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        profile_layout.addWidget(
            self.calibration_profile_combo,
            0,
            1,
            1,
            2,
        )

        self.load_calibration_profile_button = QtWidgets.QPushButton(
            "Load selected profile"
        )
        profile_layout.addWidget(
            self.load_calibration_profile_button,
            1,
            0,
            1,
            3,
        )

        profile_layout.addWidget(
            QtWidgets.QLabel("Profile name"),
            2,
            0,
        )
        self.calibration_profile_name_edit = QtWidgets.QLineEdit()
        self.calibration_profile_name_edit.setPlaceholderText(
            "e.g. Scanner_120kVp_protocol_A"
        )
        profile_layout.addWidget(
            self.calibration_profile_name_edit,
            2,
            1,
            1,
            2,
        )

        self.save_calibration_profile_button = QtWidgets.QPushButton(
            "Save current coefficients"
        )
        self.delete_calibration_profile_button = QtWidgets.QPushButton(
            "Delete selected profile"
        )
        profile_layout.addWidget(
            self.save_calibration_profile_button,
            3,
            0,
            1,
            2,
        )
        profile_layout.addWidget(
            self.delete_calibration_profile_button,
            3,
            2,
        )

        profile_note = QtWidgets.QLabel(
            "The last loaded or saved profile is restored automatically "
            "when Geo_DECT starts."
        )
        profile_note.setWordWrap(True)
        profile_layout.addWidget(
            profile_note,
            4,
            0,
            1,
            3,
        )

        stoich_layout.addWidget(profile_group)

        self.show_equations_button = QtWidgets.QPushButton(
            "Validate coefficients and show equations"
        )
        stoich_layout.addWidget(self.show_equations_button)

        self.equations_browser = QtWidgets.QTextBrowser()
        self.equations_browser.setMinimumHeight(210)
        self.equations_browser.setPlaceholderText(
            "The stoichiometric calibration equations will appear here."
        )
        stoich_layout.addWidget(self.equations_browser)

        background_group = QtWidgets.QGroupBox(
            "Retained voxels preview"
        )
        
        background_layout = QtWidgets.QGridLayout(
            background_group
        )
        
        self.background_checkbox = QtWidgets.QCheckBox(
            "Exclude background"
        )
        
        self.background_checkbox.setChecked(True)
        
        background_layout.addWidget(
            self.background_checkbox,
            0,
            0,
            1,
            2
        )
        
        self.background_threshold_label = QtWidgets.QLabel(
            "HIGH gray-level threshold"
        )

        background_layout.addWidget(
            self.background_threshold_label,
            1,
            0
        )
        
        self.background_threshold_edit = QtWidgets.QLineEdit(
            "30000"
        )
        
        background_layout.addWidget(
            self.background_threshold_edit,
            1,
            1
        )

        self.background_erosion_label = QtWidgets.QLabel(
            "Mask erosion (px)"
        )

        background_layout.addWidget(
            self.background_erosion_label,
            2,
            0
        )

        self.background_erosion_spinbox = QtWidgets.QSpinBox()
        self.background_erosion_spinbox.setRange(0, 20)
        self.background_erosion_spinbox.setValue(0)
        self.background_erosion_spinbox.setToolTip(
            "Shrinks the retained-voxel mask inward by this many "
            "pixels (per slice) before any statistics or channel is "
            "computed. Removes the thin partial-volume/mixel rim at "
            "material interfaces (sample edge, pores) that can "
            "otherwise bias Zeff/rho_e upward. Set to 0 to disable "
            "(previous behaviour). For porous samples, increase "
            "gradually and check the preview -- too much erosion "
            "will also eat genuine small pores."
        )

        background_layout.addWidget(
            self.background_erosion_spinbox,
            2,
            1
        )

        self.preview_background_button = QtWidgets.QPushButton(
            "Preview retained voxels"
        )
        
        background_layout.addWidget(
            self.preview_background_button,
            3,
            0,
            1,
            2
        )
        
        self.background_preview_label = QtWidgets.QLabel(
            "Retained voxels preview will appear here."
        )
        
        self.background_preview_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        
        self.background_preview_label.setMinimumHeight(220)
        
        self.background_preview_label.setStyleSheet(
            "border: 1px solid gray;"
        )
        self.background_preview_label.setCursor(
            QtCore.Qt.CursorShape.PointingHandCursor
        )
        self.background_preview_label.setToolTip(
            "Click the preview to open a larger image."
        )
        self.background_preview_label.installEventFilter(self)
        
        background_layout.addWidget(
            self.background_preview_label,
            4,
            0,
            1,
            2
        )
        
        self.background_preview_info = QtWidgets.QLabel("")
        
        self.background_preview_info.setWordWrap(True)
        
        background_layout.addWidget(
            self.background_preview_info,
            5,
            0,
            1,
            2
        )
        
        stoich_layout.addWidget(background_group)

        test_slice_group = QtWidgets.QGroupBox(
            "Single-slice calculation test"
        )
        test_slice_layout = QtWidgets.QGridLayout(
            test_slice_group
        )

        test_slice_layout.addWidget(
            QtWidgets.QLabel("Test slice"),
            0,
            0
        )

        self.test_slice_mode_combo = QtWidgets.QComboBox()
        self.test_slice_mode_combo.addItems([
            "Middle slice (recommended)",
            "First slice",
            "Custom slice index"
        ])
        test_slice_layout.addWidget(
            self.test_slice_mode_combo,
            0,
            1,
            1,
            2
        )

        self.custom_slice_spin = QtWidgets.QSpinBox()
        self.custom_slice_spin.setMinimum(0)
        self.custom_slice_spin.setMaximum(0)
        self.custom_slice_spin.setEnabled(False)
        test_slice_layout.addWidget(
            QtWidgets.QLabel("Custom index"),
            1,
            0
        )
        test_slice_layout.addWidget(
            self.custom_slice_spin,
            1,
            1,
            1,
            2
        )

        self.test_slice_button = QtWidgets.QPushButton(
            "Test calculation on selected image pair"
        )
        test_slice_layout.addWidget(
            self.test_slice_button,
            2,
            0,
            1,
            3
        )

        self.slice_test_results = QtWidgets.QPlainTextEdit()
        self.slice_test_results.setReadOnly(True)
        self.slice_test_results.setMinimumHeight(220)
        self.slice_test_results.setPlaceholderText(
            "Results for the selected LOW/HIGH image pair "
            "will appear here."
        )
        test_slice_layout.addWidget(
            self.slice_test_results,
            3,
            0,
            1,
            3
        )

        self.save_test_slice_button = QtWidgets.QPushButton(
            "Save test-slice maps as 32-bit TIFF..."
        )
        self.save_test_slice_button.setToolTip(
            "Save Z_eff, rho_e, the raw mu_LOW/mu_HIGH ratio, mu_LOW, "
            "mu_HIGH and the retained-voxel mask of the last single-slice "
            "test, for independent checks in ImageJ/Fiji."
        )
        test_slice_layout.addWidget(
            self.save_test_slice_button,
            4,
            0,
            1,
            3
        )

        stoich_layout.addWidget(test_slice_group)

        # ------------------------------------------------------
        # VISUALIZATION PREVIEW
        # ------------------------------------------------------
        visualization_group = QtWidgets.QGroupBox(
            "Geo_DECT visualization preview"
        )

        visualization_layout = QtWidgets.QVBoxLayout(
            visualization_group
        )

        range_layout = QtWidgets.QGridLayout()

        # Column titles
        range_layout.addWidget(
            QtWidgets.QLabel("Property"),
            0,
            0
        )

        range_layout.addWidget(
            QtWidgets.QLabel("Minimum"),
            0,
            1
        )

        range_layout.addWidget(
            QtWidgets.QLabel("Maximum"),
            0,
            2
        )

        # Zeff range
        range_layout.addWidget(
            self._html_label(
                "<i>Z</i><sub><i>eff</i></sub>"
            ),
            1,
            0
        )

        self.visualization_zeff_min_edit = (
            QtWidgets.QLineEdit()
        )

        self.visualization_zeff_max_edit = (
            QtWidgets.QLineEdit()
        )

        self.visualization_zeff_min_edit.setPlaceholderText(
            "e.g. 8"
        )

        self.visualization_zeff_max_edit.setPlaceholderText(
            "e.g. 20"
        )

        range_layout.addWidget(
            self.visualization_zeff_min_edit,
            1,
            1
        )

        range_layout.addWidget(
            self.visualization_zeff_max_edit,
            1,
            2
        )

        # rho_e range
        range_layout.addWidget(
            self._html_label(
                "<i>&rho;</i><sub><i>e</i></sub>"
            ),
            2,
            0
        )

        self.visualization_rhoe_min_edit = (
            QtWidgets.QLineEdit()
        )

        self.visualization_rhoe_max_edit = (
            QtWidgets.QLineEdit()
        )

        self.visualization_rhoe_min_edit.setPlaceholderText(
            "e.g. 0.8"
        )

        self.visualization_rhoe_max_edit.setPlaceholderText(
            "e.g. 1.6"
        )

        range_layout.addWidget(
            self.visualization_rhoe_min_edit,
            2,
            1
        )

        range_layout.addWidget(
            self.visualization_rhoe_max_edit,
            2,
            2
        )

        visualization_layout.addLayout(
            range_layout
        )

        self.visualization_auto_range_button = QtWidgets.QPushButton(
            "Detect automatic ranges from calculated slice"
        )
        self.visualization_auto_range_button.setToolTip(
            "Estimate robust property ranges from finite retained voxels. "
            "The resulting values remain editable, so the user can narrow "
            "the interval before creating the visualization."
        )
        self.visualization_auto_range_button.clicked.connect(
            self._set_automatic_visualization_ranges
        )
        visualization_layout.addWidget(
            self.visualization_auto_range_button
        )

        # Preview button
        self.create_visualization_preview_button = (
            QtWidgets.QPushButton(
                "Create visualization previews"
            )
        )

        visualization_layout.addWidget(
            self.create_visualization_preview_button
        )

        three_d_group = QtWidgets.QGroupBox(
            "Native Dragonfly 3D volumes"
        )
        three_d_layout = QtWidgets.QVBoxLayout(three_d_group)

        combined_resolution_group = QtWidgets.QGroupBox(
            "Combined colour resolution"
        )
        combined_resolution_layout = QtWidgets.QVBoxLayout(
            combined_resolution_group
        )
        self.combined_bins_70_radio = QtWidgets.QRadioButton(
            "~4,900 colours (70x70) -- fastest, safest"
        )
        self.combined_bins_100_radio = QtWidgets.QRadioButton(
            "~10,000 colours (100x100) -- medium"
        )
        self.combined_bins_122_radio = QtWidgets.QRadioButton(
            "~14,900 colours (122x122) -- finer gradient, slower"
        )
        self.combined_bins_70_radio.setChecked(True)
        combined_resolution_layout.addWidget(self.combined_bins_70_radio)
        combined_resolution_layout.addWidget(self.combined_bins_100_radio)
        combined_resolution_layout.addWidget(self.combined_bins_122_radio)
        three_d_layout.addWidget(combined_resolution_group)

        three_d_explanation = QtWidgets.QLabel(
            "Build three native Dragonfly objects: Geo_DECT, "
            "Z_eff and rho_e. The selected ranges redistribute the "
            "complete color spaces; values outside the selected "
            "range are always fully transparent."
        )
        three_d_explanation.setWordWrap(True)
        three_d_layout.addWidget(three_d_explanation)

        self.create_3d_volumes_button = QtWidgets.QPushButton(
            "Build and publish 3D volumes"
        )
        three_d_layout.addWidget(self.create_3d_volumes_button)

        self.volume_progress_bar = QtWidgets.QProgressBar()
        self.volume_progress_bar.setRange(0, 100)
        self.volume_progress_bar.setValue(0)
        three_d_layout.addWidget(self.volume_progress_bar)

        self.volume_status_label = QtWidgets.QLabel(
            "No 3D volume has been generated."
        )
        self.force_3d_coloring_button = QtWidgets.QPushButton(
            "Force 3D coloring"
        )
        three_d_layout.addWidget(self.force_3d_coloring_button)
        self.reconnect_geodect_button = QtWidgets.QPushButton(
            "Reconnect Geo_DECT volumes (after reopening a project)"
        )
        self.reconnect_geodect_button.setToolTip(
            "Re-run the same reconnect that happens automatically when "
            "this plugin opens -- use this if colours/the 3D volume "
            "did not come back after reopening a saved Dragonfly "
            "project, since running it here (well after the plugin "
            "has finished loading) can succeed where the automatic "
            "attempt did not."
        )
        three_d_layout.addWidget(self.reconnect_geodect_button)
        self.volume_status_label.setWordWrap(True)
        three_d_layout.addWidget(self.volume_status_label)

        visualization_layout.addWidget(three_d_group)

        # Information text
        self.visualization_status_label = (
            QtWidgets.QLabel(
                "Run the single-slice calculation first, "
                "then define the display ranges."
            )
        )

        self.visualization_status_label.setWordWrap(
            True
        )

        visualization_layout.addWidget(
            self.visualization_status_label
        )

        stoich_layout.addWidget(
            visualization_group
        )

        self.main_layout.addWidget(stoich_group)

        report_group = QtWidgets.QGroupBox(
            "Dataset and intensity-calibration report"
        )
        report_layout = QtWidgets.QVBoxLayout(report_group)
        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(230)
        self.status_box.setPlaceholderText(
            "Dataset and calibration information will appear here."
        )
        report_layout.addWidget(self.status_box)
        self.main_layout.addWidget(report_group)
        self.main_layout.addStretch()

        self._update_coefficient_fields()

    def _connect_signals(self):
        self.low_browse_button.clicked.connect(
            self._select_low_folder
        )
        self.high_browse_button.clicked.connect(
            self._select_high_folder
        )
        self.check_button.clicked.connect(
            self._check_datasets
        )
        self.reset_button.clicked.connect(
            self._reset_analysis
        )
        self.zeff_degree_combo.currentIndexChanged.connect(
            self._update_coefficient_fields
        )
        self.rhoe_degree_combo.currentIndexChanged.connect(
            self._update_coefficient_fields
        )
        self.show_equations_button.clicked.connect(
            self._show_equations
        )
        self.save_calibration_profile_button.clicked.connect(
            self._save_current_calibration_profile
        )
        self.load_calibration_profile_button.clicked.connect(
            self._load_selected_calibration_profile
        )
        self.delete_calibration_profile_button.clicked.connect(
            self._delete_selected_calibration_profile
        )
        self.calibration_profile_combo.currentTextChanged.connect(
            self._calibration_profile_selection_changed
        )
        self.test_slice_mode_combo.currentIndexChanged.connect(
            self._update_test_slice_controls
        )
        self.test_slice_button.clicked.connect(
            self._test_selected_tiff_pair
        )
        self.save_test_slice_button.clicked.connect(
            self._save_test_slice_tiffs
        )
        self.preview_background_button.clicked.connect(
            self._preview_background_mask
        )
        self.create_visualization_preview_button.clicked.connect(
            self._create_visualization_previews
        )
        self.create_3d_volumes_button.clicked.connect(
            self._build_and_publish_3d_volumes
        )
        self.force_3d_coloring_button.clicked.connect(
            self._apply_confirmed_3d_coloring
        )
        self.reconnect_geodect_button.clicked.connect(
            lambda checked=False: (
                self._reconnect_existing_geodect_channels(
                    force_recreate=True
                )
            )
        )

    @staticmethod
    def _html_label(html):
        label = QtWidgets.QLabel(html)
        label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        return label

    def _select_low_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select LOW-energy image folder"
        )
        if folder:
            self.low_folder_edit.setText(folder)

    def _select_high_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select HIGH-energy image folder"
        )
        if folder:
            self.high_folder_edit.setText(folder)

    @staticmethod
    def _natural_sort_key(path):
        """
        Sort filenames naturally.

        Example:
        slice2 comes before slice10.
        """
        filename = os.path.basename(path)

        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", filename)
        ]

    @classmethod
    def _tiff_files(cls, folder):
        """
        Return TIFF files without opening them.
        """
        files = [
            os.path.join(folder, filename)
            for filename in os.listdir(folder)
            if (
                os.path.isfile(
                    os.path.join(folder, filename)
                )
                and filename.lower().endswith(
                    (".tif", ".tiff")
                )
            )
        ]

        return sorted(
            files,
            key=cls._natural_sort_key
        )

    @staticmethod
    def _dicom_attributes(path):
        """Return Dragonfly DICOM attributes as name/tag lookup maps."""
        raw_attributes = LibFunctions.getDICOMAttributes(path)

        if not isinstance(raw_attributes, list) or not raw_attributes:
            raise ValueError(
                f"Dragonfly could not read DICOM attributes from: {path}"
            )

        by_name = {}
        by_tag = {}

        for item in raw_attributes:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue

            name = str(item[0])
            tag = str(item[1]).lower().replace(" ", "")
            value = item[2]

            by_name.setdefault(name, value)
            by_tag[tag] = value

        return raw_attributes, by_name, by_tag

    @staticmethod
    def _attribute_float(attributes_by_name, name, default=None):
        value = attributes_by_name.get(name, default)

        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            value = value[0] if value else default

        text = str(value).split("\\")[0].strip()

        try:
            return float(text)
        except (TypeError, ValueError):
            if default is None:
                raise ValueError(
                    f"DICOM attribute {name} is not numeric: {value}"
                )
            return float(default)

    @staticmethod
    def _attribute_int(attributes_by_name, name, default=None):
        value = MainFormGeo_dect._attribute_float(
            attributes_by_name,
            name,
            default
        )
        return None if value is None else int(value)

    @classmethod
    def _dicom_candidates(cls, folder):
        """
        Find DICOM files without opening every image.

        Files ending in .dcm are accepted directly. For an extensionless
        series, only the first file is inspected with Dragonfly. If that
        first file is a readable DICOM, all extensionless files are kept.
        """
        all_files = [
            os.path.join(folder, filename)
            for filename in os.listdir(folder)
            if os.path.isfile(
                os.path.join(folder, filename)
            )
        ]

        dicom_files = [
            path
            for path in all_files
            if path.lower().endswith(".dcm")
        ]

        if dicom_files:
            return sorted(
                dicom_files,
                key=cls._natural_sort_key
            )

        extensionless_files = [
            path
            for path in all_files
            if not os.path.splitext(path)[1]
        ]

        if not extensionless_files:
            return []

        extensionless_files = sorted(
            extensionless_files,
            key=cls._natural_sort_key
        )

        first_file = extensionless_files[0]

        try:
            attributes = LibFunctions.getDICOMAttributes(
                first_file
            )

            if isinstance(attributes, list) and attributes:
                return extensionless_files

        except Exception:
            pass

        return []

    def _detect_folder_dataset(self, folder):
        """
        Detect TIFF or DICOM without reading the entire stack.

        TIFF detection stops immediately when TIFF files are found.
        DICOM headers are read later only from the first LOW and HIGH files.
        """
        tiff_files = self._tiff_files(folder)

        if tiff_files:
            return "tiff", tiff_files

        dicom_files = self._dicom_candidates(folder)

        if dicom_files:
            return "dicom", dicom_files

        raise ValueError(
            "No TIFF or readable DICOM images were found."
        )

    @classmethod
    def _read_dicom(cls, path):
        """
        Read one DICOM image through Dragonfly's native API.

        Dragonfly returns stored pixel values. HU are reconstructed from
        RescaleSlope and RescaleIntercept. Identity rescaling leaves values
        unchanged. A Modality LUT is rejected explicitly because silently
        ignoring it could produce incorrect quantitative values.
        """
        raw_attributes, by_name, by_tag = cls._dicom_attributes(path)

        if "ModalityLUTSequence" in by_name:
            raise ValueError(
                "This DICOM uses a Modality LUT. Geo_DECT currently supports "
                "CT DICOM series expressed through RescaleSlope and "
                "RescaleIntercept, but will not ignore a Modality LUT."
            )

        stored_pixels = np.asarray(
            LibFunctions.getDICOMPixelArray(path)
        )

        if stored_pixels.size == 0:
            raise ValueError(
                f"Dragonfly returned an empty DICOM pixel array: {path}"
            )

        slope = cls._attribute_float(
            by_name, "RescaleSlope", 1.0
        )
        intercept = cls._attribute_float(
            by_name, "RescaleIntercept", 0.0
        )
        rescale_type = str(
            by_name.get("RescaleType", "")
        ).strip()
        modality = str(
            by_name.get("Modality", "")
        ).strip()

        hu = compute_hu(
            stored_pixels,
            slope,
            intercept,
        )
        mu = compute_mu(
            hu,
            {"mode": "dicom_hu"},
        )

        if slope == 1.0 and intercept == 0.0:
            interpretation = (
                "Identity rescale: stored values are used unchanged "
                "as HU."
            )
        else:
            interpretation = (
                "HU reconstructed automatically from "
                "RescaleSlope/RescaleIntercept."
            )

        calibration = {
            "mode": "dicom_hu",
            "rescale_slope": slope,
            "rescale_intercept": intercept,
            "interpretation": interpretation,
            "modality": modality,
            "rescale_type": rescale_type,
            "instance_number": cls._attribute_int(
                by_name, "InstanceNumber", None
            ),
            "stored_min": float(np.min(stored_pixels)),
            "stored_max": float(np.max(stored_pixels)),
            "hu_min": float(np.min(hu)),
            "hu_max": float(np.max(hu)),
            "attributes": raw_attributes,
            "attributes_by_name": by_name,
            "attributes_by_tag": by_tag,
        }

        return stored_pixels, hu, mu, calibration

    @staticmethod
    def _extract_number(text, names):
        if not text:
            return None
        number_pattern = (
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        )
        for name in names:
            pattern = rf"\b{re.escape(name)}\b\s*[:=]\s*{number_pattern}"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _detect_tiff_calibration(self, tiff_path):
        with tf.TiffFile(tiff_path) as tif:
            page = tif.pages[0]
            description = page.description or ""
            imagej_metadata = tif.imagej_metadata or {}
            metadata_parts = [description]
            for tag in page.tags.values():
                try:
                    metadata_parts.append(f"{tag.name}={tag.value}")
                except Exception:
                    pass
            metadata_text = "\n".join(metadata_parts)

        slope = self._extract_number(
            metadata_text, ["slope", "rescale slope", "rescaleslope"]
        )
        offset = self._extract_number(
            metadata_text,
            ["offset", "intercept", "rescale intercept", "rescaleintercept"]
        )
        if slope is not None and offset is not None:
            return {"mode": "slope_offset", "slope": slope, "offset": offset}

        physical_min = self._extract_number(
            metadata_text,
            [
                "physical_min", "physical min", "physicalminimum",
                "physical minimum", "intensity_min", "intensity min",
                "minimum intensity", "calibrated_min", "calibrated min"
            ]
        )
        physical_max = self._extract_number(
            metadata_text,
            [
                "physical_max", "physical max", "physicalmaximum",
                "physical maximum", "intensity_max", "intensity max",
                "maximum intensity", "calibrated_max", "calibrated max"
            ]
        )
        if physical_min is not None and physical_max is not None:
            return {
                "mode": "physical_min_max",
                "min": physical_min,
                "max": physical_max,
            }

        return {
            "mode": "manual_required",
            "imagej_min": imagej_metadata.get("min"),
            "imagej_max": imagej_metadata.get("max"),
        }

    @staticmethod
    def _format_calibration_report(label, calibration):
        mode = calibration["mode"]
        if mode == "slope_offset":
            return [
                f"{label} calibration",
                "  Mode: slope/offset from metadata",
                f"  Slope: {calibration['slope']}",
                f"  Offset: {calibration['offset']}",
                "  Formula:",
                "  mu = gray level * slope + offset",
            ]
        if mode == "physical_min_max":
            return [
                f"{label} calibration",
                "  Mode: physical min/max from metadata",
                f"  Physical min: {calibration['min']}",
                f"  Physical max: {calibration['max']}",
                "  Formula:",
                "  mu = (gray level / 65535) * (max - min) + min",
            ]

        lines = [
            f"{label} calibration",
            "  Mode: manual physical min/max required",
            "  Formula:",
            "  mu = (gray level / 65535) * (max - min) + min",
        ]
        if calibration.get("imagej_min") is not None:
            lines.extend([
                "  ImageJ display values found and ignored:",
                f"  Display min: {calibration['imagej_min']}",
                f"  Display max: {calibration['imagej_max']}",
            ])
        return lines

    def _update_manual_fields(self):
        tiff_mode = (
            self.low_format == "tiff"
            and self.high_format == "tiff"
        )

        self.manual_calibration_group.setEnabled(tiff_mode)

        low_manual = (
            tiff_mode
            and self.low_calibration is not None
            and self.low_calibration["mode"] == "manual_required"
        )
        high_manual = (
            tiff_mode
            and self.high_calibration is not None
            and self.high_calibration["mode"] == "manual_required"
        )

        self.low_min_edit.setEnabled(low_manual)
        self.low_max_edit.setEnabled(low_manual)
        self.high_min_edit.setEnabled(high_manual)
        self.high_max_edit.setEnabled(high_manual)

        if not low_manual:
            self.low_min_edit.clear()
            self.low_max_edit.clear()
        if not high_manual:
            self.high_min_edit.clear()
            self.high_max_edit.clear()

    def _update_format_dependent_controls(self):
        if self.low_format == "dicom":
            self.background_threshold_label.setText(
                "HIGH HU threshold"
            )
            self.background_threshold_edit.setText("-950")
        else:
            self.background_threshold_label.setText(
                "HIGH gray-level threshold"
            )
            self.background_threshold_edit.setText("30000")
    
    def _check_datasets(self):
        self.status_box.clear()
        low_folder = self.low_folder_edit.text().strip()
        high_folder = self.high_folder_edit.text().strip()

        try:
            if not os.path.isdir(low_folder):
                raise ValueError("The LOW folder is not valid.")
            if not os.path.isdir(high_folder):
                raise ValueError("The HIGH folder is not valid.")

            self.low_format, self.low_files = (
                self._detect_folder_dataset(low_folder)
            )
            self.high_format, self.high_files = (
                self._detect_folder_dataset(high_folder)
            )

            if self.low_format != self.high_format:
                raise ValueError(
                    "LOW and HIGH folders contain different image formats. "
                    "Both must be TIFF or both DICOM."
                )

            common_slice_count = min(
                len(self.low_files),
                len(self.high_files)
            )
            self.custom_slice_spin.setMaximum(
                max(0, common_slice_count - 1)
            )
            self.custom_slice_spin.setValue(
                common_slice_count // 2
            )

            # Auto-detect voxel spacing and show it explicitly for the
            # user to confirm -- a silently pre-filled field is exactly
            # how the original 100x/200x spacing bugs went unnoticed
            # for as long as they did. Every "Check" run surfaces this,
            # even if the override field already has something in it,
            # so a stale or wrong value never just sits there unseen.
            try:
                detected_spacing_m = self._auto_detect_spacing_xyz_m()
                detected_x_um, detected_y_um, detected_z_um = (
                    v * 1.0e6 for v in detected_spacing_m
                )

                # The fallback path (no ImageJ metadata, no XResolution/
                # YResolution/ResolutionUnit tags at all) always lands
                # on exactly 1 metre per axis -- no real dataset has
                # this as its true spacing, so this triple match is a
                # reliable signal that NOTHING was actually found in
                # the file, rather than a real (if implausible)
                # detected value the user should merely double-check.
                no_metadata_found = (
                    abs(detected_x_um - 1.0e6) < 1.0
                    and abs(detected_y_um - 1.0e6) < 1.0
                    and abs(detected_z_um - 1.0e6) < 1.0
                )

                current_override = self.voxel_size_override_edit.text().strip()

                if no_metadata_found:
                    proposed_value = current_override
                    confirmed_text, accepted = QtWidgets.QInputDialog.getText(
                        self,
                        "Geo_DECT — Voxel size required",
                        "This dataset's TIFF files carry NO physical "
                        "spacing metadata at all (no ImageJ calibration, "
                        "no XResolution/YResolution/ResolutionUnit tags) "
                        "-- there is nothing here to auto-detect from.\n\n"
                        "Enter the true voxel size (µm), taken from your "
                        "scan/acquisition software, before building -- "
                        "one value for an isotropic voxel, or 'X, Y, Z' "
                        "for an anisotropic one:",
                        QtWidgets.QLineEdit.EchoMode.Normal,
                        proposed_value,
                    )
                    if accepted and confirmed_text.strip():
                        self.voxel_size_override_edit.setText(
                            confirmed_text.strip()
                        )
                else:
                    axis_values = (
                        detected_x_um, detected_y_um, detected_z_um
                    )
                    spread = (
                        (max(axis_values) / min(axis_values))
                        if min(axis_values) > 0 else float("inf")
                    )
                    mismatch_warning = ""
                    # Medical DICOM volumes are anisotropic by design
                    # (slice spacing Z typically 2-10x the in-plane
                    # pixel spacing) and their spacing comes from
                    # standard, reliable tags -- a large X/Y/Z spread is
                    # expected there, not a symptom of misread metadata.
                    # The warning is therefore kept for TIFF only.
                    if self.low_format != "dicom" and spread > 1.5:
                        mismatch_warning = (
                            "\n\nWarning: X/Y/Z detected spacing differ "
                            "by more than 50% from each other -- this is "
                            "exactly the kind of mismatch that has "
                            "previously come from misread file metadata. "
                            "Double-check this value (e.g. against what "
                            "your acquisition software reports) before "
                            "building."
                        )

                    # Propose the full X, Y, Z triple (collapsed to a
                    # single number only when it is truly isotropic).
                    # Proposing the X/Y/Z MEAN here -- as before -- is
                    # what silently turned anisotropic DICOM voxels
                    # into isotropic ones: once confirmed, that mean
                    # landed in the override field, which always wins
                    # over auto-detection at Build time.
                    proposed_value = (
                        current_override
                        if current_override
                        else self._format_spacing_um(axis_values)
                    )

                    confirmed_text, accepted = QtWidgets.QInputDialog.getText(
                        self,
                        "Geo_DECT — Confirm voxel size",
                        "Auto-detected from file metadata:\n"
                        "X: {} µm   Y: {} µm   Z: {} µm{}\n\n"
                        "Voxel size to use (µm): one value for an "
                        "isotropic voxel, or 'X, Y, Z' for an anisotropic "
                        "one -- check it against your scan/acquisition "
                        "software and edit below if it looks wrong:".format(
                            self._format_number(detected_x_um),
                            self._format_number(detected_y_um),
                            self._format_number(detected_z_um),
                            mismatch_warning,
                        ),
                        QtWidgets.QLineEdit.EchoMode.Normal,
                        proposed_value,
                    )
                    if accepted and confirmed_text.strip():
                        self.voxel_size_override_edit.setText(
                            confirmed_text.strip()
                        )
            except Exception:
                pass

            if self.low_format == "tiff":
                low_first = tf.imread(self.low_files[0])
                high_first = tf.imread(self.high_files[0])

                self.low_calibration = (
                    self._detect_tiff_calibration(
                        self.low_files[0]
                    )
                )
                self.high_calibration = (
                    self._detect_tiff_calibration(
                        self.high_files[0]
                    )
                )

                calibration_report = [
                    "",
                    *self._format_calibration_report(
                        "LOW", self.low_calibration
                    ),
                    "",
                    *self._format_calibration_report(
                        "HIGH", self.high_calibration
                    ),
                ]

            else:
                (
                    low_first,
                    _,
                    _,
                    self.low_calibration
                ) = self._read_dicom(self.low_files[0])
                (
                    high_first,
                    _,
                    _,
                    self.high_calibration
                ) = self._read_dicom(self.high_files[0])

                calibration_report = [
                    "",
                    "LOW DICOM interpretation",
                    f"  {self.low_calibration['interpretation']}",
                    f"  Modality: {self.low_calibration['modality'] or 'not specified'}",
                    f"  RescaleSlope: {self.low_calibration['rescale_slope']}",
                    f"  RescaleIntercept: {self.low_calibration['rescale_intercept']}",
                    f"  Stored range: {self.low_calibration['stored_min']:.8g} to "
                    f"{self.low_calibration['stored_max']:.8g}",
                    f"  HU range: {self.low_calibration['hu_min']:.8g} to "
                    f"{self.low_calibration['hu_max']:.8g}",
                    "  mu_relative = HU / 1000 + 1",
                    "",
                    "HIGH DICOM interpretation",
                    f"  {self.high_calibration['interpretation']}",
                    f"  Modality: {self.high_calibration['modality'] or 'not specified'}",
                    f"  RescaleSlope: {self.high_calibration['rescale_slope']}",
                    f"  RescaleIntercept: {self.high_calibration['rescale_intercept']}",
                    f"  Stored range: {self.high_calibration['stored_min']:.8g} to "
                    f"{self.high_calibration['stored_max']:.8g}",
                    f"  HU range: {self.high_calibration['hu_min']:.8g} to "
                    f"{self.high_calibration['hu_max']:.8g}",
                    "  mu_relative = HU / 1000 + 1",
                    "",
                    "Electron-density outputs:",
                    "  rho_e / rho_e,water is calculated directly.",
                    "  Absolute rho_e is also calculated automatically",
                    f"  using rho_e,water = {WATER_ELECTRON_DENSITY_CM3:.6e} electrons/cm^3.",
                ]

            self._update_manual_fields()
            self._update_format_dependent_controls()

            report = [
                "LOW dataset",
                f"  Format: {self.low_format.upper()}",
                f"  Images: {len(self.low_files)}",
                f"  First image shape: {low_first.shape}",
                f"  Data type: {low_first.dtype}",
                "",
                "HIGH dataset",
                f"  Format: {self.high_format.upper()}",
                f"  Images: {len(self.high_files)}",
                f"  First image shape: {high_first.shape}",
                f"  Data type: {high_first.dtype}",
                "",
            ]

            report.append(
                "Slice count: OK"
                if len(self.low_files) == len(self.high_files)
                else "Slice count: ERROR — LOW and HIGH differ"
            )
            report.append(
                "Image dimensions: OK"
                if low_first.shape == high_first.shape
                else "Image dimensions: ERROR — LOW and HIGH differ"
            )
            report.append(
                "Data type: OK"
                if low_first.dtype == high_first.dtype
                else "Data type: WARNING — LOW and HIGH differ"
            )
            report.extend(calibration_report)

            self.status_box.setPlainText("\n".join(report))

        except Exception as error:
            self.low_format = None
            self.high_format = None
            self.low_files = []
            self.high_files = []
            self.low_calibration = None
            self.high_calibration = None

            for widget in (
                self.low_min_edit,
                self.low_max_edit,
                self.high_min_edit,
                self.high_max_edit,
            ):
                widget.setEnabled(False)

            self.status_box.setPlainText(
                f"Dataset check failed:\n{error}"
            )
            
            QtWidgets.QMessageBox.critical(
                self,
                "Geo_DECT — Dataset check failed",
                str(error)
            )

    @staticmethod
    def _normalized_profile_name(name):
        """Return a safe, readable profile name."""
        name = " ".join(str(name).strip().split())
        if not name:
            raise ValueError(
                "Enter a name for the stoichiometric calibration profile."
            )
        if len(name) > 120:
            raise ValueError(
                "The calibration profile name must contain at most 120 characters."
            )
        return name

    def _load_calibration_profiles_from_disk(self):
        """Read saved calibration profiles without blocking plugin startup."""
        self.calibration_profiles = {}
        self.last_calibration_profile = ""

        if not os.path.isfile(self.calibration_profiles_path):
            return

        try:
            with open(
                self.calibration_profiles_path,
                "r",
                encoding="utf-8",
            ) as file_handle:
                payload = json.load(file_handle)

            if not isinstance(payload, dict):
                raise ValueError("The profile file root must be a JSON object.")

            raw_profiles = payload.get("profiles", {})
            if not isinstance(raw_profiles, dict):
                raise ValueError("The profiles entry must be a JSON object.")

            profiles = {}
            for raw_name, raw_profile in raw_profiles.items():
                try:
                    name = self._normalized_profile_name(raw_name)
                    if not isinstance(raw_profile, dict):
                        continue

                    zeff_degree = int(raw_profile["zeff_degree"])
                    rhoe_degree = int(raw_profile["rhoe_degree"])
                    zeff_coefficients = [
                        float(value)
                        for value in raw_profile["zeff_coefficients"]
                    ]
                    rhoe_coefficients = [
                        float(value)
                        for value in raw_profile["rhoe_coefficients"]
                    ]
                    numerator_mode = str(
                        raw_profile.get("rhoe_numerator_mode", "low")
                    ).lower()

                    if zeff_degree < 0 or zeff_degree > 5:
                        continue
                    if rhoe_degree < 0 or rhoe_degree > 5:
                        continue
                    if len(zeff_coefficients) != zeff_degree + 1:
                        continue
                    if len(rhoe_coefficients) != rhoe_degree + 1:
                        continue
                    if numerator_mode not in {"low", "high", "mean"}:
                        numerator_mode = "low"

                    profiles[name] = {
                        "zeff_degree": zeff_degree,
                        "rhoe_degree": rhoe_degree,
                        "zeff_coefficients": zeff_coefficients,
                        "rhoe_coefficients": rhoe_coefficients,
                        "rhoe_numerator_mode": numerator_mode,
                    }
                except Exception:
                    continue

            self.calibration_profiles = profiles
            last_name = str(
                payload.get("last_profile", "")
            ).strip()
            if last_name in profiles:
                self.last_calibration_profile = last_name

        except Exception as error:
            self.calibration_profiles = {}
            self.last_calibration_profile = ""
            QtWidgets.QMessageBox.warning(
                self,
                "Geo_DECT — Calibration profiles",
                "The saved stoichiometric calibration profiles could not "
                "be read. Geo_DECT will continue with an empty profile list."
                "\n\n{}".format(error),
            )

    def _write_calibration_profiles_to_disk(self):
        """Write profiles atomically to reduce the risk of corruption."""
        folder = os.path.dirname(self.calibration_profiles_path)
        os.makedirs(folder, exist_ok=True)

        payload = {
            "version": 1,
            "last_profile": self.last_calibration_profile,
            "profiles": self.calibration_profiles,
        }
        temporary_path = self.calibration_profiles_path + ".tmp"

        with open(
            temporary_path,
            "w",
            encoding="utf-8",
        ) as file_handle:
            json.dump(
                payload,
                file_handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            file_handle.write("\n")

        os.replace(
            temporary_path,
            self.calibration_profiles_path,
        )

    def _refresh_calibration_profile_combo(self, selected_name=None):
        """Refresh the profile list while preserving the requested selection."""
        if not hasattr(self, "calibration_profile_combo"):
            return

        if selected_name is None:
            selected_name = (
                self.calibration_profile_combo.currentText().strip()
                or self.last_calibration_profile
            )

        self.calibration_profile_combo.blockSignals(True)
        self.calibration_profile_combo.clear()
        self.calibration_profile_combo.addItem("")

        for name in sorted(
            self.calibration_profiles,
            key=str.casefold,
        ):
            self.calibration_profile_combo.addItem(name)

        index = self.calibration_profile_combo.findText(
            selected_name
        )
        self.calibration_profile_combo.setCurrentIndex(
            index if index >= 0 else 0
        )
        self.calibration_profile_combo.blockSignals(False)

    def _calibration_profile_selection_changed(self, name):
        name = str(name).strip()
        if name:
            self.calibration_profile_name_edit.setText(name)

    def _current_calibration_profile_data(self):
        """Validate the current controls and serialize their values."""
        zeff_degree = int(
            self.zeff_degree_combo.currentText()
        )
        rhoe_degree = int(
            self.rhoe_degree_combo.currentText()
        )
        zeff_coefficients = self._read_coefficients(
            self.zeff_coeff_edits,
            zeff_degree,
            "Zeff",
        )
        rhoe_coefficients = self._read_coefficients(
            self.rhoe_coeff_edits,
            rhoe_degree,
            "rho_e",
        )
        _html, numerator_mode = (
            self._selected_rhoe_numerator_html()
        )

        return {
            "zeff_degree": zeff_degree,
            "rhoe_degree": rhoe_degree,
            "zeff_coefficients": zeff_coefficients,
            "rhoe_coefficients": rhoe_coefficients,
            "rhoe_numerator_mode": numerator_mode,
        }

    def _apply_calibration_profile(self, name, show_message=True):
        """Populate all stoichiometric controls from one saved profile."""
        if name not in self.calibration_profiles:
            raise ValueError(
                "The selected calibration profile no longer exists."
            )

        profile = self.calibration_profiles[name]
        zeff_degree = int(profile["zeff_degree"])
        rhoe_degree = int(profile["rhoe_degree"])

        self.zeff_degree_combo.setCurrentIndex(zeff_degree)
        self.rhoe_degree_combo.setCurrentIndex(rhoe_degree)
        self._update_coefficient_fields()

        for edit in self.zeff_coeff_edits:
            edit.clear()
        for edit in self.rhoe_coeff_edits:
            edit.clear()

        for index, value in enumerate(
            profile["zeff_coefficients"]
        ):
            self.zeff_coeff_edits[index].setText(
                self._format_number(float(value))
            )

        for index, value in enumerate(
            profile["rhoe_coefficients"]
        ):
            self.rhoe_coeff_edits[index].setText(
                self._format_number(float(value))
            )

        numerator_mode = profile.get(
            "rhoe_numerator_mode",
            "low",
        )
        if numerator_mode == "high":
            self.rhoe_mu_high_radio.setChecked(True)
        elif numerator_mode == "mean":
            self.rhoe_mu_mean_radio.setChecked(True)
        else:
            self.rhoe_mu_low_radio.setChecked(True)

        self.calibration_profile_name_edit.setText(name)
        self._refresh_calibration_profile_combo(name)

        self.last_calibration_profile = name
        self._write_calibration_profiles_to_disk()

        self._show_equations()

        if show_message:
            QtWidgets.QMessageBox.information(
                self,
                "Geo_DECT — Calibration loaded",
                "Stoichiometric calibration profile loaded:\n\n{}".format(
                    name
                ),
            )

    def _load_last_calibration_profile(self):
        """Restore the last used profile when the plugin opens."""
        name = self.last_calibration_profile
        if not name or name not in self.calibration_profiles:
            return

        try:
            self._apply_calibration_profile(
                name,
                show_message=False,
            )
        except Exception:
            pass

    def _load_selected_calibration_profile(self):
        name = self.calibration_profile_combo.currentText().strip()
        if not name:
            QtWidgets.QMessageBox.information(
                self,
                "Geo_DECT — Calibration profiles",
                "Select a saved calibration profile first.",
            )
            return

        try:
            self._apply_calibration_profile(name)
        except Exception as error:
            QtWidgets.QMessageBox.critical(
                self,
                "Geo_DECT — Calibration load failed",
                str(error),
            )

    def _save_current_calibration_profile(self):
        try:
            name = self._normalized_profile_name(
                self.calibration_profile_name_edit.text()
            )
            profile = self._current_calibration_profile_data()

            if name in self.calibration_profiles:
                answer = QtWidgets.QMessageBox.question(
                    self,
                    "Geo_DECT — Replace calibration profile",
                    "A profile named {!r} already exists.\n\n"
                    "Replace it with the current coefficients?".format(name),
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    return

            self.calibration_profiles[name] = profile
            self.last_calibration_profile = name
            self._write_calibration_profiles_to_disk()
            self._refresh_calibration_profile_combo(name)
            self.calibration_profile_name_edit.setText(name)

            QtWidgets.QMessageBox.information(
                self,
                "Geo_DECT — Calibration saved",
                "Stoichiometric calibration profile saved:\n\n{}\n\n"
                "It will be loaded automatically the next time "
                "Geo_DECT starts.".format(name),
            )

        except Exception as error:
            QtWidgets.QMessageBox.critical(
                self,
                "Geo_DECT — Calibration save failed",
                str(error),
            )

    def _delete_selected_calibration_profile(self):
        name = self.calibration_profile_combo.currentText().strip()
        if not name:
            QtWidgets.QMessageBox.information(
                self,
                "Geo_DECT — Calibration profiles",
                "Select a saved calibration profile first.",
            )
            return

        answer = QtWidgets.QMessageBox.question(
            self,
            "Geo_DECT — Delete calibration profile",
            "Delete the stoichiometric calibration profile {!r}?".format(
                name
            ),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            self.calibration_profiles.pop(name, None)
            if self.last_calibration_profile == name:
                self.last_calibration_profile = ""
            self._write_calibration_profiles_to_disk()
            self._refresh_calibration_profile_combo()
            self.calibration_profile_name_edit.clear()

            QtWidgets.QMessageBox.information(
                self,
                "Geo_DECT — Calibration deleted",
                "Calibration profile deleted:\n\n{}".format(name),
            )

        except Exception as error:
            QtWidgets.QMessageBox.critical(
                self,
                "Geo_DECT — Calibration deletion failed",
                str(error),
            )

    def _update_coefficient_fields(self):
        zeff_degree = int(self.zeff_degree_combo.currentText())
        rhoe_degree = int(self.rhoe_degree_combo.currentText())

        for degree, edit in enumerate(self.zeff_coeff_edits):
            enabled = degree <= zeff_degree
            edit.setEnabled(enabled)
            if not enabled:
                edit.clear()

        for degree, edit in enumerate(self.rhoe_coeff_edits):
            enabled = degree <= rhoe_degree
            edit.setEnabled(enabled)
            if not enabled:
                edit.clear()

    @staticmethod
    def _read_coefficients(edits, degree, label):
        coefficients = []
        for power in range(degree + 1):
            text = edits[power].text().strip()
            if not text:
                raise ValueError(
                    f"{label}: coefficient for power {power} is empty."
                )
            try:
                value = float(text)
            except ValueError as error:
                raise ValueError(
                    f"{label}: coefficient for power {power} is not a valid number."
                ) from error
            coefficients.append(value)
        return coefficients

    def _read_ratio_bounds(self):
        """Read the mu_LOW/mu_HIGH plausible-range fields.

        Either field left blank disables that side of the check
        (matches compute_zeff's ratio_min=None/ratio_max=None
        convention). Raises ValueError on a non-numeric entry, same
        style as _read_coefficients.
        """
        def _parse(edit, label):
            text = edit.text().strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError as error:
                raise ValueError(
                    f"Ratio {label}: not a valid number."
                ) from error

        ratio_min = _parse(self.ratio_min_edit, "minimum")
        ratio_max = _parse(self.ratio_max_edit, "maximum")

        if (
            ratio_min is not None
            and ratio_max is not None
            and ratio_max <= ratio_min
        ):
            raise ValueError(
                "Ratio maximum must be greater than ratio minimum."
            )

        return ratio_min, ratio_max

    def _rhoe_is_relative_to_water(self):
        """True when relative-to-water conversion is allowed at all."""
        if getattr(self, "low_format", None) == "dicom":
            return True
        return bool(self.rhoe_relative_to_water_check.isChecked())

    def _rhoe_needs_water_conversion(self, rhoe_values):
        """Decide from the data whether rho_e must be multiplied by
        rho_e,water.

        Calibration profiles can return either rho_e / rho_e,water
        (values of a few units) or absolute rho_e (values ~1e23-1e24).
        Converting already-absolute values would multiply them by 3.343
        a second time, so the conversion is applied only when the
        computed values are clearly relative (median below 1e3).
        """
        if not self._rhoe_is_relative_to_water():
            return False
        values = np.asarray(rhoe_values, dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size == 0:
            return None
        return bool(np.median(values) < 1.0e3)

    def _read_denoise_settings(self):
        """Return (method, strength) from the noise-reduction controls."""
        method = self.denoise_method_combo.currentData() or "none"
        strength = float(self.denoise_strength_spin.value())
        return method, strength

    @staticmethod
    def _describe_denoise(method, strength):
        if method == "gaussian":
            return "Gaussian, mask-aware, sigma = {:g} px".format(strength)
        if method == "median":
            size = max(3, int(round(strength)))
            if size % 2 == 0:
                size += 1
            return "Median, window = {} px".format(size)
        return "none (raw voxels)"

    def _denoise_mu_pair(self, mu_low, mu_high, mask, method, strength):
        """Apply the same noise reduction to mu_LOW and mu_HIGH."""
        if method == "none":
            return mu_low, mu_high
        return (
            denoise_mu(mu_low, method, strength, valid_mask=mask),
            denoise_mu(mu_high, method, strength, valid_mask=mask),
        )

    @staticmethod
    def _rhoe_report_units(rhoe_display_scale):
        """Units used for rho_e in the PDF reports.

        The stored rho_e arrays are already divided by rhoe_display_scale
        (1e23 for absolute electron density), so the reports keep those
        numbers and state the multiplier explicitly in every label,
        e.g. "x10^23 e-.cm-3", instead of mixing 7.7 and 7.7e+23.
        Returns (multiplier_to_apply, unit_text).
        """
        scale = float(rhoe_display_scale or 1.0)
        if scale == 1.0:
            return 1.0, "relative units"
        exponent = int(round(np.log10(scale)))
        superscript = str(exponent).translate(
            str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")
        )
        return (
            scale / (10.0 ** exponent),
            "\u00d710{} e\u207b\u00b7cm\u207b\u00b3".format(superscript),
        )

    @staticmethod
    def _format_number(value):
        if value == 0:
            return "0"
        absolute = abs(value)
        if absolute >= 1e5 or absolute < 1e-4:
            return f"{value:.6e}"
        return f"{value:.10g}"

    @staticmethod
    def _format_range_input(value, scientific=False):
        """Format editable range fields without excessive digits."""
        value = float(value)
        return f"{value:.1e}" if scientific else f"{value:.1f}"

    @staticmethod
    def _format_zeff_readout(value):
        """Format Z_eff for the combined redistribution readout with a
        fixed, compact number of decimals -- display only, the
        underlying selection/computation keeps full precision."""
        return f"{float(value):.2f}"

    @staticmethod
    def _format_tick_value(value, scientific=False):
        """Format legend ticks using one decimal digit."""
        value = float(value)
        if scientific:
            if value == 0.0:
                return "0.0"
            exponent = int(np.floor(np.log10(abs(value))))
            mantissa = value / (10.0 ** exponent)
            superscript = str(exponent).translate(
                str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
            )
            return f"{mantissa:.2f}×10{superscript}"
        return f"{value:.1f}"

    @staticmethod
    def _draw_html_centered(painter, html, rectangle):
        """Draw centered rich text with italics and full subscripts."""
        document = QtGui.QTextDocument()
        document.setDocumentMargin(0)
        document.setDefaultStyleSheet(
            "body { color: #000000; text-align: center; }"
        )
        document.setHtml(html)
        document.setTextWidth(rectangle.width())

        painter.save()
        content_height = document.size().height()
        painter.translate(
            rectangle.left(),
            rectangle.top()
            + max(0.0, (rectangle.height() - content_height) / 2.0)
        )
        document.drawContents(
            painter,
            QtCore.QRectF(0, 0, rectangle.width(), rectangle.height())
        )
        painter.restore()

    @classmethod
    def _polynomial_html(cls, coefficients, variable_html):
        terms = []
        for power, coefficient in enumerate(coefficients):
            if coefficient == 0:
                continue
            number = cls._format_number(abs(coefficient))
            if power == 0:
                core = number
            elif power == 1:
                core = f"{number}&nbsp;{variable_html}"
            else:
                core = f"{number}&nbsp;{variable_html}<sup>{power}</sup>"

            if not terms:
                terms.append(f"&minus;{core}" if coefficient < 0 else core)
            else:
                sign = "&minus;" if coefficient < 0 else "+"
                terms.append(f"&nbsp;{sign}&nbsp;{core}")
        return "".join(terms) if terms else "0"

    def _validate_manual_physical_ranges(self):
        if self.low_calibration is None or self.high_calibration is None:
            raise ValueError("Check the LOW and HIGH datasets first.")

        if self.low_format == "dicom":
            return

        self.low_calibration = self._complete_manual_calibration(
            self.low_calibration,
            self.low_min_edit,
            self.low_max_edit,
            "LOW",
        )
        self.high_calibration = self._complete_manual_calibration(
            self.high_calibration,
            self.high_min_edit,
            self.high_max_edit,
            "HIGH",
        )

    @staticmethod
    def _complete_manual_calibration(calibration, min_edit, max_edit, label):
        if calibration["mode"] != "manual_required":
            return calibration

        min_text = min_edit.text().strip()
        max_text = max_edit.text().strip()
        if not min_text or not max_text:
            raise ValueError(f"{label}: physical min and max are required.")

        try:
            physical_min = float(min_text)
            physical_max = float(max_text)
        except ValueError as error:
            raise ValueError(
                f"{label}: physical min and max must be valid numbers."
            ) from error

        if physical_max <= physical_min:
            raise ValueError(
                f"{label}: physical max must be greater than physical min."
            )

        return {
            "mode": "physical_min_max",
            "source": "manual",
            "min": physical_min,
            "max": physical_max,
        }

    def _selected_rhoe_numerator_html(self):
        mu_low = "<i>&mu;</i><sub><i>LOW</i></sub>"
        mu_high = "<i>&mu;</i><sub><i>HIGH</i></sub>"
        if self.rhoe_mu_low_radio.isChecked():
            return mu_low, "low"
        if self.rhoe_mu_high_radio.isChecked():
            return mu_high, "high"
        return f"({mu_low}&nbsp;+&nbsp;{mu_high})&nbsp;/&nbsp;2", "mean"

    def _show_equations(self):
        try:
            self._validate_manual_physical_ranges()

            zeff_degree = int(self.zeff_degree_combo.currentText())
            rhoe_degree = int(self.rhoe_degree_combo.currentText())
            zeff_coefficients = self._read_coefficients(
                self.zeff_coeff_edits, zeff_degree, "Zeff"
            )
            rhoe_coefficients = self._read_coefficients(
                self.rhoe_coeff_edits, rhoe_degree, "rho_e"
            )

            mu_low = "<i>&mu;</i><sub><i>LOW</i></sub>"
            mu_high = "<i>&mu;</i><sub><i>HIGH</i></sub>"
            zeff = "<i>Z</i><sub><i>eff</i></sub>"
            rhoe = "<i>&rho;</i><sub><i>e</i></sub>"
            ratio = f"({mu_low}&nbsp;/&nbsp;{mu_high})"

            zeff_polynomial = self._polynomial_html(
                zeff_coefficients, ratio
            )
            rhoe_denominator = self._polynomial_html(
                rhoe_coefficients, zeff
            )
            numerator_html, numerator_mode = (
                self._selected_rhoe_numerator_html()
            )

            if getattr(
                self, "test_rhoe_converted", self.low_format == "dicom"
            ):
                rhoe_left = (
                    f"{rhoe}&nbsp;/&nbsp;"
                    f"{rhoe}<sub>water</sub>"
                )
                density_note = (
                    "<p><b>Relative-to-water convention:</b> "
                    "electron density is relative to water. "
                    "Absolute electron density is obtained by multiplying "
                    "by the electron density of water.</p>"
                )
            else:
                rhoe_left = rhoe
                density_note = ""

            html = f"""
            <div style="font-size: 12pt;">
                <p><b>Validated stoichiometric equations</b></p>
                <p>{zeff}&nbsp;=&nbsp;{zeff_polynomial}</p>
                <p>
                    {rhoe_left}&nbsp;=&nbsp;
                    {numerator_html}&nbsp;/&nbsp;({rhoe_denominator})
                </p>
                {density_note}
                <hr>
                <p>
                    <b>Selected polynomial degrees</b><br>
                    {zeff}: {zeff_degree}<br>
                    {rhoe}: {rhoe_degree}
                </p>
                <p>
                    <b>Numerator selected for {rhoe}</b><br>
                    {numerator_html}
                </p>
            </div>
            """
            self.equations_browser.setHtml(html)

            self.zeff_degree = zeff_degree
            self.rhoe_degree = rhoe_degree
            self.zeff_coefficients = zeff_coefficients
            self.rhoe_coefficients = rhoe_coefficients
            self.rhoe_numerator_mode = numerator_mode

        except Exception as error:
            self.equations_browser.setPlainText(
                f"Equation validation failed:\n{error}"
            )

    @staticmethod
    def _array_statistics(name, array):
        finite = array[np.isfinite(array)]

        if finite.size == 0:
            return [name, "  No finite values were calculated."]

        return [
            name,
            f"  Shape: {array.shape}",
            f"  Data type: {array.dtype}",
            f"  Finite pixels: {finite.size}",
            f"  Minimum: {np.min(finite):.8g}",
            f"  Maximum: {np.max(finite):.8g}",
            f"  Mean: {np.mean(finite):.8g}",
            f"  Median: {np.median(finite):.8g}",
        ]

    def _update_test_slice_controls(self):
        custom_mode = (
            self.test_slice_mode_combo.currentText()
            == "Custom slice index"
        )
        self.custom_slice_spin.setEnabled(custom_mode)

    def _selected_test_slice_index(self):
        if not self.low_files or not self.high_files:
            raise ValueError(
                "Check the datasets and calibration first."
            )

        common_slice_count = min(
            len(self.low_files),
            len(self.high_files)
        )

        if common_slice_count == 0:
            raise ValueError(
                "No common LOW/HIGH image pairs are available."
            )

        mode = self.test_slice_mode_combo.currentText()

        if mode == "First slice":
            return 0

        if mode == "Custom slice index":
            index = self.custom_slice_spin.value()
        else:
            index = common_slice_count // 2

        if index >= common_slice_count:
            raise ValueError(
                "The selected slice index is outside the "
                "available LOW/HIGH range."
            )

        return index

    @staticmethod
    def _normalize_to_uint8(array):
        array = np.asarray(array, dtype=np.float32)
        finite = array[np.isfinite(array)]

        if finite.size == 0:
            return np.zeros(array.shape, dtype=np.uint8)

        low = np.percentile(finite, 1.0)
        high = np.percentile(finite, 99.0)

        if high <= low:
            high = low + 1.0

        normalized = np.clip(
            (array - low) / (high - low),
            0.0,
            1.0
        )

        return (normalized * 255.0).astype(np.uint8)

    def _background_mask(self, high_gray, high_hu=None):
        if not self.background_checkbox.isChecked():
            return None, None

        threshold_text = (
            self.background_threshold_edit.text().strip()
        )

        try:
            threshold = float(threshold_text)
        except ValueError as error:
            raise ValueError(
                "Background threshold must be numeric."
            ) from error

        if self.high_format == "dicom":
            if high_hu is None:
                raise ValueError(
                    "HIGH HU values are required for DICOM background exclusion."
                )
            mask = high_hu > threshold
        else:
            mask = high_gray > threshold

        # Shrink the retained mask inward (per slice, 2D) to drop the
        # thin partial-volume/mixel rim at the sample's outer surface
        # and at internal pore/grain interfaces. Those voxels blend
        # signal from two different materials and, once run through
        # the calibration curve, can land on a physically meaningless
        # but plausible-looking Zeff/rho_e -- eroding them out here is
        # cheaper and safer than trying to filter them after the fact.
        erosion_px = int(self.background_erosion_spinbox.value())
        if erosion_px > 0 and np.any(mask):
            mask = ndimage.binary_erosion(
                mask, iterations=erosion_px
            )

        return mask, threshold

    def _preview_background_mask(self):
        """Preview retained voxels with a purple overlay."""
        try:
            slice_index = self._selected_test_slice_index()
            high_path = self.high_files[slice_index]

            if self.high_format == "tiff":
                high_gray = tf.imread(high_path)
                high_hu = None
                display_source = high_gray
            else:
                high_gray, high_hu, _, _ = self._read_dicom(
                    high_path
                )
                display_source = high_hu

            mask, threshold = self._background_mask(
                high_gray,
                high_hu
            )

            if mask is None:
                raise ValueError(
                    "Background exclusion is disabled."
                )

            grayscale = self._normalize_to_uint8(display_source)
            rgb = np.stack(
                [grayscale, grayscale, grayscale],
                axis=-1
            ).astype(np.float32)

            overlay_color = np.array(
                [140.0, 70.0, 180.0],
                dtype=np.float32
            )
            alpha = 0.45

            rgb[mask] = (
                (1.0 - alpha) * rgb[mask]
                + alpha * overlay_color
            )

            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb = np.ascontiguousarray(rgb)
            height, width, _ = rgb.shape

            image = QtGui.QImage(
                rgb.data,
                width,
                height,
                rgb.strides[0],
                QtGui.QImage.Format.Format_RGB888
            ).copy()

            self._full_preview_pixmap = QtGui.QPixmap.fromImage(image)

            small_pixmap = self._full_preview_pixmap.scaled(
                260,
                260,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            self.background_preview_label.setPixmap(small_pixmap)

            retained_pixels = int(np.count_nonzero(mask))
            total_pixels = int(mask.size)
            excluded_pixels = total_pixels - retained_pixels
            retained_percent = 100.0 * retained_pixels / total_pixels
            excluded_percent = 100.0 * excluded_pixels / total_pixels

            threshold_unit = (
                "HU"
                if self.high_format == "dicom"
                else "gray level"
            )

            self.background_preview_info.setText(
                f"Slice: {slice_index}\n"
                f"HIGH file: {os.path.basename(high_path)}\n"
                f"Threshold: {threshold:g} {threshold_unit}\n\n"
                f"Retained voxels:\n"
                f"{retained_pixels:,} ({retained_percent:.2f}%)\n\n"
                f"Excluded voxels:\n"
                f"{excluded_pixels:,} ({excluded_percent:.2f}%)\n\n"
                "Click the image to enlarge it."
            )

        except Exception as error:
            self._full_preview_pixmap = None
            self.background_preview_label.clear()
            self.background_preview_label.setText(
                "Retained voxels preview failed."
            )
            self.background_preview_info.setText(str(error))

    def eventFilter(self, watched, event):
        if (
            watched is self.background_preview_label
            and event.type()
            == QtCore.QEvent.Type.MouseButtonRelease
            and event.button()
            == QtCore.Qt.MouseButton.LeftButton
        ):
            self._show_enlarged_preview()
            return True

        return super().eventFilter(watched, event)

    def _show_enlarged_preview(self):
        """Open a resizable, scrollable preview window."""
        if self._full_preview_pixmap is None:
            QtWidgets.QMessageBox.information(
                self,
                "Geo_DECT — Preview",
                "Create the retained-voxels preview first."
            )
            return

        self._show_enlarged_pixmap_dialog(
            "Geo_DECT — Retained voxels",
            self._full_preview_pixmap,
            "Geo_DECT_retained_voxels.png",
        )

    def _reset_analysis(self):
        """Clear all inputs, cached arrays, reports and previews."""
        self.low_format = None
        self.high_format = None
        self.low_calibration = None
        self.high_calibration = None
        self.low_files = []
        self.high_files = []

        for attribute_name in (
            "test_slice_index",
            "test_mu_low",
            "test_mu_high",
            "test_zeff",
            "test_rhoe",
            "test_rhoe_absolute",
            "test_rhoe_converted",
            "test_valid_mask",
            "zeff_degree",
            "rhoe_degree",
            "zeff_coefficients",
            "rhoe_coefficients",
            "rhoe_numerator_mode",
        ):
            if hasattr(self, attribute_name):
                delattr(self, attribute_name)

        self.low_folder_edit.clear()
        self.high_folder_edit.clear()
        self.low_min_edit.clear()
        self.low_max_edit.clear()
        self.high_min_edit.clear()
        self.high_max_edit.clear()

        self.manual_calibration_group.setEnabled(True)
        for widget in (
            self.low_min_edit,
            self.low_max_edit,
            self.high_min_edit,
            self.high_max_edit,
        ):
            widget.setEnabled(False)

        self.zeff_degree_combo.setCurrentIndex(2)
        self.rhoe_degree_combo.setCurrentIndex(2)

        for edit in self.zeff_coeff_edits + self.rhoe_coeff_edits:
            edit.clear()

        self.rhoe_mu_low_radio.setChecked(True)
        self.test_slice_mode_combo.setCurrentIndex(0)
        self.custom_slice_spin.setMaximum(0)
        self.custom_slice_spin.setValue(0)
        self.custom_slice_spin.setEnabled(False)

        self.background_checkbox.setChecked(True)
        self.background_threshold_label.setText(
            "HIGH gray-level threshold"
        )
        self.background_threshold_edit.setText("30000")

        self.visualization_zeff_min_edit.clear()
        self.visualization_zeff_max_edit.clear()
        self.visualization_rhoe_min_edit.clear()
        self.visualization_rhoe_max_edit.clear()
        self.visualization_status_label.setText(
            "Run the single-slice calculation first, "
            "then define the display ranges."
        )
        self.volume_progress_bar.setValue(0)
        self.volume_status_label.setText(
            "No 3D volume has been generated."
        )
        self.geodect_3d_channels = {}
        self.geodect_3d_visuals = {}
        self.geodect_3d_clip_boxes = {}
        self.geodect_3d_legends = {}
        self.geodect_3d_scalar_collections = {}
        self.geodect_3d_luts = {}
        self.geodect_3d_display_settings = {}
        self.geodect_3d_scene_layout = None
        self.geodect_3d_current_view = None
        if self._display_range_dock is not None:
            try:
                self._display_range_dock.close()
                self._display_range_dock.deleteLater()
            except Exception:
                pass
            self._display_range_dock = None
            self._display_range_controls = {}

        self.status_box.clear()
        self.equations_browser.clear()
        self.slice_test_results.clear()
        self.background_preview_info.clear()
        self.background_preview_label.clear()
        self.background_preview_label.setText(
            "Retained voxels preview will appear here."
        )

        self._full_preview_pixmap = None

        self.visualization_zeff_min_edit.clear()
        self.visualization_zeff_max_edit.clear()
        self.visualization_rhoe_min_edit.clear()
        self.visualization_rhoe_max_edit.clear()
        self.visualization_status_label.setText(
            "Run the single-slice calculation first, "
            "then define the display ranges."
        )

        if self._preview_dialog is not None:
            self._preview_dialog.close()
            self._preview_dialog = None

        if self._visualization_dialog is not None:
            self._visualization_dialog.close()
            self._visualization_dialog = None

        self._update_coefficient_fields()

    def closeEvent(self, event):
        self._reset_analysis()
        super().closeEvent(event)

    @staticmethod
    def _finite_display_range(array, valid_mask=None, positive_only=False):
        """Return the finite range of retained physically valid voxels."""
        values = np.asarray(array, dtype=np.float64)
        selection = np.isfinite(values)

        if valid_mask is not None:
            selection &= np.asarray(valid_mask, dtype=bool)
        if positive_only:
            selection &= values > 0.0

        finite = values[selection]
        if finite.size == 0:
            raise ValueError(
                "No positive finite values were found inside the "
                "retained-voxel mask."
            )

        minimum = float(np.min(finite))
        maximum = float(np.max(finite))

        if maximum <= minimum:
            maximum = minimum + max(abs(minimum) * 0.01, 1.0)

        return minimum, maximum

    def _visualization_rhoe_data(self):
        """Return the rho_e array and display metadata for the current input."""
        if not hasattr(self, "test_rhoe"):
            raise ValueError(
                "Run the single-slice calculation first."
            )

        if (
            getattr(self, "test_rhoe_converted", False)
            and hasattr(self, "test_rhoe_absolute")
        ):
            return (
                np.asarray(self.test_rhoe_absolute),
                "ρₑ",
                "e⁻·cm⁻³",
                "absolute electron density"
            )

        rhoe = np.asarray(self.test_rhoe)
        finite = rhoe[np.isfinite(rhoe)]

        # TIFF calibration may already produce absolute electron density.
        # Use magnitude only to select a display unit; do not rescale data.
        if finite.size and float(np.nanmedian(np.abs(finite))) > 1.0e12:
            return (
                rhoe,
                "ρₑ",
                "e⁻·cm⁻³",
                "electron density"
            )

        return (
            rhoe,
            "ρₑ",
            "relative units",
            "relative electron density"
        )

    def _set_automatic_visualization_ranges(self):
        """Populate positive ranges and warn about non-physical voxels."""
        try:
            if not hasattr(self, "test_zeff"):
                raise ValueError(
                    "Run the single-slice calculation first."
                )

            zeff = np.asarray(self.test_zeff)
            rhoe, _symbol, rhoe_unit, _description = (
                self._visualization_rhoe_data()
            )
            valid_mask = getattr(self, "test_valid_mask", None)

            common = np.isfinite(zeff) & np.isfinite(rhoe)
            if valid_mask is not None:
                common &= np.asarray(valid_mask, dtype=bool)

            negative_zeff = int(
                np.count_nonzero(common & (zeff <= 0.0))
            )
            negative_rhoe = int(
                np.count_nonzero(common & (rhoe <= 0.0))
            )

            zeff_minimum, zeff_maximum = self._finite_display_range(
                zeff,
                valid_mask=valid_mask,
                positive_only=True
            )
            rhoe_minimum, rhoe_maximum = self._finite_display_range(
                rhoe,
                valid_mask=valid_mask,
                positive_only=True
            )

            self.visualization_zeff_min_edit.setText(
                self._format_range_input(zeff_minimum)
            )
            self.visualization_zeff_max_edit.setText(
                self._format_range_input(zeff_maximum)
            )

            scientific_rhoe = rhoe_unit == "e⁻·cm⁻³"
            self.visualization_rhoe_min_edit.setText(
                self._format_range_input(
                    rhoe_minimum,
                    scientific=scientific_rhoe
                )
            )
            self.visualization_rhoe_max_edit.setText(
                self._format_range_input(
                    rhoe_maximum,
                    scientific=scientific_rhoe
                )
            )

            message = (
                "Automatic positive ranges detected. "
                "The limits remain editable."
            )

            if negative_zeff or negative_rhoe:
                message += (
                    " Warning: non-physical values were found inside "
                    "the retained-voxel mask."
                )
                warning_box = QtWidgets.QMessageBox(self)
                warning_box.setIcon(
                    QtWidgets.QMessageBox.Icon.Warning
                )
                warning_box.setWindowTitle(
                    "Geo_DECT — Check background exclusion"
                )
                warning_box.setTextFormat(
                    QtCore.Qt.TextFormat.RichText
                )
                warning_box.setText(
                    "<div style='font-size:11pt;'>"
                    "Negative or zero property values were detected inside "
                    "the retained-voxel mask."
                    "<br><br>"
                    "<b><i>Z<sub>eff</sub></i></b>: "
                    f"{negative_zeff:,} voxels"
                    "<br>"
                    "<b><i>ρ<sub>e</sub></i></b>: "
                    f"{negative_rhoe:,} voxels"
                    "<br><br>"
                    "These voxels were excluded from automatic range "
                    "detection. The gray-level threshold may be too "
                    "permissive. Also check LOW/HIGH registration and "
                    "calibration."
                    "</div>"
                )
                warning_box.exec()

            self.visualization_status_label.setText(message)

        except Exception as error:
            QtWidgets.QMessageBox.warning(
                self,
                "Geo_DECT — Automatic range detection",
                str(error)
            )

    @staticmethod
    def _read_visualization_range(minimum_edit, maximum_edit, array, label):
        """Read one requested range, using robust data limits when blank."""
        automatic_minimum, automatic_maximum = (
            MainFormGeo_dect._finite_display_range(
                array,
                positive_only=True
            )
        )

        minimum_text = minimum_edit.text().strip()
        maximum_text = maximum_edit.text().strip()

        try:
            minimum = (
                float(minimum_text)
                if minimum_text
                else automatic_minimum
            )
            maximum = (
                float(maximum_text)
                if maximum_text
                else automatic_maximum
            )
        except ValueError as error:
            raise ValueError(
                f"{label}: minimum and maximum must be numeric."
            ) from error

        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError(
                f"{label}: minimum and maximum must be finite."
            )

        if maximum <= minimum:
            raise ValueError(
                f"{label}: maximum must be greater than minimum."
            )

        minimum_edit.setText(
            MainFormGeo_dect._format_number(minimum)
        )
        maximum_edit.setText(
            MainFormGeo_dect._format_number(maximum)
        )

        return minimum, maximum

    @staticmethod
    def _normalized_property(array, minimum, maximum):
        array = np.asarray(array, dtype=np.float32)
        normalized = (array - minimum) / (maximum - minimum)
        return np.clip(normalized, 0.0, 1.0)

    @staticmethod
    def _apply_display_context(rgb, array, minimum, maximum):
        """Mask out-of-range and non-finite pixels (always transparent)."""
        output = np.asarray(rgb, dtype=np.uint8).copy()
        values = np.asarray(array)
        finite = np.isfinite(values)

        outside = (
            (~finite)
            | (values < minimum)
            | (values > maximum)
        )
        output[outside] = (0, 0, 0)

        return output

    @staticmethod
    def _create_scalar_preview_rgb(
        array,
        minimum,
        maximum,
        palette,
    ):
        """Create a cold Zeff or warm rho_e RGB preview."""
        values = np.asarray(array, dtype=np.float32)
        finite = np.isfinite(values)

        # Non-finite values must never reach the palette-index calculation:
        # converting NaN to an integer produces -2147483648, which is not a
        # valid palette index.
        safe_values = np.where(finite, values, minimum)

        normalized = MainFormGeo_dect._normalized_property(
            safe_values,
            minimum,
            maximum
        )

        stops = np.asarray(palette, dtype=np.float32)
        position = normalized * (len(stops) - 1)
        lower = np.floor(position).astype(np.int32)
        lower = np.clip(lower, 0, len(stops) - 1)
        upper = np.clip(lower + 1, 0, len(stops) - 1)
        fraction = (position - lower)[..., None]

        rgb = (
            stops[lower] * (1.0 - fraction)
            + stops[upper] * fraction
        )
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb[~finite] = (0, 0, 0)

        return MainFormGeo_dect._apply_display_context(
            rgb,
            values,
            minimum,
            maximum,
        )

    @staticmethod
    def _lab_to_rgb_uint8(lightness, a_channel, b_channel):
        """Convert CIE L*a*b* arrays to an sRGB uint8 image."""
        lightness = np.asarray(lightness, dtype=np.float64)
        a_channel = np.asarray(a_channel, dtype=np.float64)
        b_channel = np.asarray(b_channel, dtype=np.float64)

        fy = (lightness + 16.0) / 116.0
        fx = fy + a_channel / 500.0
        fz = fy - b_channel / 200.0

        delta = 6.0 / 29.0

        def inverse_f(component):
            return np.where(
                component > delta,
                component ** 3,
                3.0 * delta ** 2 * (component - 4.0 / 29.0)
            )

        x = 0.95047 * inverse_f(fx)
        y = 1.00000 * inverse_f(fy)
        z = 1.08883 * inverse_f(fz)

        red_linear = (
            3.2404542 * x
            - 1.5371385 * y
            - 0.4985314 * z
        )
        green_linear = (
            -0.9692660 * x
            + 1.8760108 * y
            + 0.0415560 * z
        )
        blue_linear = (
            0.0556434 * x
            - 0.2040259 * y
            + 1.0572252 * z
        )

        linear = np.stack(
            [red_linear, green_linear, blue_linear],
            axis=-1
        )
        linear = np.clip(linear, 0.0, 1.0)

        rgb = np.where(
            linear <= 0.0031308,
            12.92 * linear,
            1.055 * np.power(linear, 1.0 / 2.4) - 0.055
        )

        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _create_geodect_preview_rgb(
        zeff,
        rhoe,
        zeff_minimum,
        zeff_maximum,
        rhoe_minimum,
        rhoe_maximum,
    ):
        """Create the combined Geo_DECT CIE L*a*b* visualization."""
        zeff_normalized = MainFormGeo_dect._normalized_property(
            zeff,
            zeff_minimum,
            zeff_maximum
        )
        rhoe_normalized = MainFormGeo_dect._normalized_property(
            rhoe,
            rhoe_minimum,
            rhoe_maximum
        )

        lightness = np.full(
            zeff_normalized.shape,
            68.0,
            dtype=np.float64
        )
        a_channel = -55.0 + 110.0 * zeff_normalized
        b_channel = -55.0 + 110.0 * rhoe_normalized

        rgb = MainFormGeo_dect._lab_to_rgb_uint8(
            lightness,
            a_channel,
            b_channel
        )

        finite = np.isfinite(zeff) & np.isfinite(rhoe)
        zeff_below = zeff < zeff_minimum
        zeff_above = zeff > zeff_maximum
        rhoe_below = rhoe < rhoe_minimum
        rhoe_above = rhoe > rhoe_maximum

        outside = (
            (~finite)
            | zeff_below
            | zeff_above
            | rhoe_below
            | rhoe_above
        )
        rgb[outside] = (0, 0, 0)

        return rgb

    @staticmethod
    def _numpy_rgb_to_pixmap(rgb):
        """Convert an HxWx3 uint8 NumPy array to a detached QPixmap."""
        rgb = np.ascontiguousarray(
            np.asarray(rgb, dtype=np.uint8)
        )

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                "Visualization output must be an H x W x 3 RGB array."
            )

        height, width, _ = rgb.shape
        image = QtGui.QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QtGui.QImage.Format.Format_RGB888
        ).copy()

        return QtGui.QPixmap.fromImage(image)

    def _save_pixmap_as_image(self, pixmap, default_name):
        """Prompt for a path and save a QPixmap as a PNG file."""
        file_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save image",
            default_name,
            "PNG images (*.png)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".png"):
            file_path += ".png"
        if not pixmap.save(file_path, "PNG"):
            QtWidgets.QMessageBox.critical(
                self,
                "Save image failed",
                "The image could not be saved to:\n{}".format(file_path),
            )

    def _show_enlarged_pixmap_dialog(
        self, title, pixmap, default_save_name, max_size=900
    ):
        """Generic resizable, scrollable preview window for any pixmap,
        with a button to save it as a PNG (for papers/presentations)."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(max_size, max_size)

        layout = QtWidgets.QVBoxLayout(dialog)
        scroll_area = QtWidgets.QScrollArea(dialog)
        scroll_area.setWidgetResizable(True)

        image_label = QtWidgets.QLabel()
        image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        target_size = QtCore.QSize(max_size - 40, max_size - 40)
        if (
            pixmap.width() > target_size.width()
            or pixmap.height() > target_size.height()
        ):
            shown_pixmap = pixmap.scaled(
                target_size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        else:
            shown_pixmap = pixmap
        image_label.setPixmap(shown_pixmap)
        image_label.setMinimumSize(shown_pixmap.size())

        scroll_area.setWidget(image_label)
        layout.addWidget(scroll_area)

        buttons_row = QtWidgets.QHBoxLayout()
        save_button = QtWidgets.QPushButton("Save as image...")
        save_button.clicked.connect(
            lambda: self._save_pixmap_as_image(pixmap, default_save_name)
        )
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        buttons_row.addWidget(save_button)
        buttons_row.addWidget(close_button)
        layout.addLayout(buttons_row)

        dialog.exec()

    def _create_labeled_colorbar_pixmap(
        self, colorbar_rgb, range_minimum, range_maximum,
        unit_suffix="", tick_count=4, target_width=300, gradient_height=34,
        tick_font_size=11,
    ):
        """Compose the plain gradient array into a labeled colorbar: a
        few evenly spaced tick marks with compact (1-decimal) values,
        the selected min/max in bold, everything else in normal weight.
        Built directly at the size it will actually be displayed at
        (not scaled down afterward), so the text stays legible without
        needing to enlarge the widget."""
        gradient_pixmap = self._numpy_rgb_to_pixmap(colorbar_rgb).scaled(
            target_width, gradient_height,
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

        bold_font = QtGui.QFont()
        bold_font.setPointSize(tick_font_size)
        bold_font.setBold(True)
        normal_font = QtGui.QFont()
        normal_font.setPointSize(tick_font_size)

        tick_height = 6
        label_metrics = QtGui.QFontMetrics(bold_font)
        label_area_height = label_metrics.height() + 6
        unit_area_height = (
            (label_metrics.height() + 2) if unit_suffix else 0
        )
        total_height = (
            gradient_height + tick_height
            + label_area_height + unit_area_height
        )

        composed = QtGui.QPixmap(target_width, total_height)
        composed.fill(QtCore.Qt.GlobalColor.white)

        painter = QtGui.QPainter(composed)
        painter.drawPixmap(0, 0, gradient_pixmap)

        width = target_width
        tick_count = max(int(tick_count), 2)

        for index in range(tick_count):
            fraction = index / (tick_count - 1)
            x_position = int(round(fraction * (width - 1)))
            value = range_minimum + fraction * (range_maximum - range_minimum)
            is_endpoint = index == 0 or index == tick_count - 1

            painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black, 1))
            painter.drawLine(
                x_position, gradient_height,
                x_position, gradient_height + tick_height,
            )

            painter.setFont(bold_font if is_endpoint else normal_font)
            text = self._format_tick_value(value)
            metrics = QtGui.QFontMetrics(
                bold_font if is_endpoint else normal_font
            )
            text_width = metrics.horizontalAdvance(text)
            text_x = min(
                max(x_position - text_width // 2, 0),
                width - text_width,
            )
            painter.drawText(
                text_x,
                gradient_height + tick_height + metrics.ascent() + 2,
                text,
            )

        if unit_suffix:
            painter.setFont(normal_font)
            metrics = QtGui.QFontMetrics(normal_font)
            unit_text = "({})".format(unit_suffix)
            unit_width = metrics.horizontalAdvance(unit_text)
            painter.drawText(
                max((width - unit_width) // 2, 0),
                total_height - 4,
                unit_text,
            )

        painter.end()
        return composed

    @staticmethod
    def _create_scalar_colorbar_pixmap(
        minimum,
        maximum,
        palette,
        title,
        scientific_ticks=False,
        width=190,
        height=440
    ):
        """Create a vertical scalar colorbar with readable scientific labels."""
        canvas = QtGui.QPixmap(width, height)
        canvas.fill(QtCore.Qt.GlobalColor.white)

        painter = QtGui.QPainter(canvas)
        painter.setRenderHint(
            QtGui.QPainter.RenderHint.Antialiasing,
            True
        )

        # Leave enough vertical space between the title and the bar.
        bar_left = 24
        bar_top = 68
        bar_width = 38
        bar_height = height - bar_top - 28

        gradient = QtGui.QLinearGradient(
            0,
            bar_top + bar_height,
            0,
            bar_top
        )
        for index, color in enumerate(palette):
            fraction = index / max(1, len(palette) - 1)
            gradient.setColorAt(
                fraction,
                QtGui.QColor(
                    int(color[0]),
                    int(color[1]),
                    int(color[2])
                )
            )

        painter.fillRect(
            QtCore.QRectF(
                bar_left,
                bar_top,
                bar_width,
                bar_height
            ),
            gradient
        )
        painter.setPen(
            QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.2)
        )
        painter.drawRect(
            bar_left,
            bar_top,
            bar_width,
            bar_height
        )

        MainFormGeo_dect._draw_html_centered(
            painter,
            title,
            QtCore.QRectF(0, 4, width, 52)
        )

        font = painter.font()
        font.setPointSize(9)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black))

        tick_count = 6
        for index in range(tick_count):
            fraction = index / (tick_count - 1)
            y = bar_top + bar_height * (1.0 - fraction)
            value = minimum + fraction * (maximum - minimum)

            painter.drawLine(
                bar_left + bar_width,
                int(y),
                bar_left + bar_width + 7,
                int(y)
            )
            painter.drawText(
                QtCore.QRectF(
                    bar_left + bar_width + 12,
                    y - 11,
                    width - bar_left - bar_width - 16,
                    22
                ),
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignVCenter,
                MainFormGeo_dect._format_tick_value(
                    value,
                    scientific=scientific_ticks
                )
            )

        painter.end()
        return canvas

    @staticmethod
    def _create_geodect_legend_pixmap(
        zeff_minimum,
        zeff_maximum,
        rhoe_minimum,
        rhoe_maximum,
        size=420
    ):
        """Create the numeric 2D Geo_DECT legend."""
        margin_left = 128
        margin_right = 24
        margin_top = 30
        margin_bottom = 92
        grid_width = size - margin_left - margin_right
        grid_height = size - margin_top - margin_bottom

        sample_size = 256
        zeff_normalized = np.tile(
            np.linspace(0.0, 1.0, sample_size),
            (sample_size, 1)
        )
        rhoe_normalized = np.tile(
            np.linspace(1.0, 0.0, sample_size)[:, None],
            (1, sample_size)
        )

        lightness = np.full(
            (sample_size, sample_size),
            68.0,
            dtype=np.float64
        )
        rgb = MainFormGeo_dect._lab_to_rgb_uint8(
            lightness,
            -55.0 + 110.0 * zeff_normalized,
            -55.0 + 110.0 * rhoe_normalized
        )
        grid_pixmap = MainFormGeo_dect._numpy_rgb_to_pixmap(
            rgb
        ).scaled(
            grid_width,
            grid_height,
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )

        canvas = QtGui.QPixmap(size, size)
        canvas.fill(QtCore.Qt.GlobalColor.white)

        painter = QtGui.QPainter(canvas)
        painter.setRenderHint(
            QtGui.QPainter.RenderHint.Antialiasing,
            True
        )
        painter.drawPixmap(
            margin_left,
            margin_top,
            grid_pixmap
        )
        painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black))
        painter.drawRect(
            margin_left,
            margin_top,
            grid_width,
            grid_height
        )

        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)

        tick_count = 5
        for index in range(tick_count):
            fraction = index / (tick_count - 1)

            x = margin_left + fraction * grid_width
            zeff_value = (
                zeff_minimum
                + fraction * (zeff_maximum - zeff_minimum)
            )
            painter.drawLine(
                int(x),
                margin_top + grid_height,
                int(x),
                margin_top + grid_height + 6
            )
            painter.drawText(
                QtCore.QRectF(
                    x - 38,
                    margin_top + grid_height + 8,
                    76,
                    20
                ),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                MainFormGeo_dect._format_tick_value(zeff_value)
            )

            y = margin_top + grid_height * (1.0 - fraction)
            rhoe_value = (
                rhoe_minimum
                + fraction * (rhoe_maximum - rhoe_minimum)
            )
            painter.drawLine(
                margin_left - 6,
                int(y),
                margin_left,
                int(y)
            )
            painter.drawText(
                QtCore.QRectF(
                    0,
                    y - 10,
                    margin_left - 10,
                    20
                ),
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter,
                MainFormGeo_dect._format_tick_value(rhoe_value, scientific=True)
            )

        MainFormGeo_dect._draw_html_centered(
            painter,
            "<div style='color:#000000; font-size:17pt; "
            "font-weight:700; text-align:center;'>"
            "<i>Z<sub>eff</sub></i></div>",
            QtCore.QRectF(
                margin_left,
                margin_top + grid_height + 42,
                grid_width,
                42
            )
        )

        # Rotate around the vertical midpoint. The larger left margin gives
        # rho_e the same visual separation from its tick values as Z_eff.
        painter.save()
        painter.translate(38, margin_top + grid_height / 2)
        painter.rotate(-90)
        MainFormGeo_dect._draw_html_centered(
            painter,
            "<div style='color:#000000; font-size:16pt; "
            "font-weight:700; text-align:center;'>"
            "<i>ρ<sub>e</sub></i> "
            "<span style='font-size:11pt;'>"
            "(<i>e</i><sup>−</sup>·<i>cm</i><sup>−3</sup>)"
            "</span></div>",
            QtCore.QRectF(
                -grid_height / 2,
                -34,
                grid_height,
                68
            )
        )
        painter.restore()

        painter.end()
        return canvas

    @staticmethod
    def _visualization_image_panel(title, image_pixmap, legend_pixmap):
        """Build one titled image-and-legend column."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title_label = QtWidgets.QLabel(title)
        title_label.setTextFormat(
            QtCore.Qt.TextFormat.RichText
        )
        title_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        title_label.setStyleSheet(
            "font-size: 17px;"
        )
        layout.addWidget(title_label)

        image_label = QtWidgets.QLabel()
        image_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        image_label.setStyleSheet(
            "background: black; border: 1px solid gray;"
        )
        image_label.setPixmap(
            image_pixmap.scaled(
                520,
                520,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
        )
        image_label.setMinimumSize(360, 360)
        layout.addWidget(image_label, 1)

        legend_label = QtWidgets.QLabel()
        legend_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        legend_label.setPixmap(legend_pixmap)
        layout.addWidget(legend_label)

        return panel

    def _create_visualization_previews(self):
        """Create the first real Geo_DECT, Zeff and rho_e preview window."""
        try:
            if not hasattr(self, "test_zeff") or not hasattr(
                self, "test_rhoe"
            ):
                raise ValueError(
                    "Run the single-slice calculation first."
                )

            zeff = np.asarray(self.test_zeff)
            (
                rhoe,
                rhoe_symbol,
                rhoe_unit,
                rhoe_description
            ) = self._visualization_rhoe_data()

            if zeff.shape != rhoe.shape:
                raise ValueError(
                    "Zeff and rho_e arrays have different dimensions."
                )

            zeff_minimum, zeff_maximum = (
                self._read_visualization_range(
                    self.visualization_zeff_min_edit,
                    self.visualization_zeff_max_edit,
                    zeff,
                    "Zeff"
                )
            )
            rhoe_minimum, rhoe_maximum = (
                self._read_visualization_range(
                    self.visualization_rhoe_min_edit,
                    self.visualization_rhoe_max_edit,
                    rhoe,
                    "rho_e"
                )
            )

            zeff_palette = list(ZEFF_PALETTE_RGB255)
            rhoe_palette = list(RHOE_PALETTE_RGB255)

            geodect_rgb = create_geodect_rgb(
                zeff,
                rhoe,
                zeff_minimum,
                zeff_maximum,
                rhoe_minimum,
                rhoe_maximum,
            )
            zeff_rgb = create_zeff_rgb(
                zeff,
                zeff_minimum,
                zeff_maximum,
            )
            rhoe_rgb = create_rhoe_rgb(
                rhoe,
                rhoe_minimum,
                rhoe_maximum,
            )

            geodect_pixmap = self._numpy_rgb_to_pixmap(
                geodect_rgb
            )
            zeff_pixmap = self._numpy_rgb_to_pixmap(
                zeff_rgb
            )
            rhoe_pixmap = self._numpy_rgb_to_pixmap(
                rhoe_rgb
            )

            geodect_legend = self._create_geodect_legend_pixmap(
                zeff_minimum,
                zeff_maximum,
                rhoe_minimum,
                rhoe_maximum
            )
            zeff_colorbar = self._create_scalar_colorbar_pixmap(
                zeff_minimum,
                zeff_maximum,
                zeff_palette,
                "<div style='color:#000000; font-size:13pt; "
                "font-weight:700; text-align:center;'>"
                "<i>Z<sub>eff</sub></i></div>"
                "<div style='color:#000000; font-size:9pt; "
                "text-align:center;'>effective atomic number "
                "(dimensionless)</div>",
                scientific_ticks=False
            )
            rhoe_colorbar = self._create_scalar_colorbar_pixmap(
                rhoe_minimum,
                rhoe_maximum,
                rhoe_palette,
                "<div style='color:#000000; font-size:13pt; "
                "font-weight:700; text-align:center;'>"
                "<i>ρ<sub>e</sub></i></div>"
                "<div style='color:#000000; font-size:9pt; "
                "text-align:center;'>electron density "
                "(<i>e</i><sup>−</sup>·<i>cm</i><sup>−3</sup>)"
                "</div>",
                scientific_ticks=True
            )

            if self._visualization_dialog is not None:
                self._visualization_dialog.close()

            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(
                "Geo_DECT — Single-slice visualization"
            )
            dialog.resize(1550, 930)
            dialog.setMinimumSize(900, 650)

            dialog_layout = QtWidgets.QVBoxLayout(dialog)

            scientific_rhoe = rhoe_unit == "e⁻·cm⁻³"
            information = QtWidgets.QLabel(
                "<div style='font-size:13pt; font-weight:700;'>"
                f"Slice {getattr(self, 'test_slice_index', '?')} "
                "&nbsp;|&nbsp; "
                "<i>Z<sub>eff</sub></i>: "
                f"{self._format_tick_value(zeff_minimum)}–"
                f"{self._format_tick_value(zeff_maximum)} "
                "&nbsp;|&nbsp; "
                "<i>ρ<sub>e</sub></i>: "
                f"{self._format_tick_value(rhoe_minimum, scientific_rhoe)}–"
                f"{self._format_tick_value(rhoe_maximum, scientific_rhoe)} "
                "<i>e</i><sup>−</sup>·<i>cm</i><sup>−3</sup>"
                "</div>"
            )
            information.setTextFormat(
                QtCore.Qt.TextFormat.RichText
            )
            information.setWordWrap(True)
            information.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
            )
            dialog_layout.addWidget(information)

            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            scroll_area.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )

            content = QtWidgets.QWidget()
            content_layout = QtWidgets.QHBoxLayout(content)
            content_layout.setContentsMargins(8, 8, 8, 8)
            content_layout.setSpacing(12)

            content_layout.addWidget(
                self._visualization_image_panel(
                    "<b>Geo_DECT</b> — combined "
                    "<b><i>Z<sub>eff</sub></i></b> and "
                    "<b><i>ρ<sub>e</sub></i></b>",
                    geodect_pixmap,
                    geodect_legend
                )
            )
            content_layout.addWidget(
                self._visualization_image_panel(
                    "<b><i>Z<sub>eff</sub></i></b> — "
                    "<span style='font-size:14px;'>effective atomic number "
                    "(dimensionless)</span>",
                    zeff_pixmap,
                    zeff_colorbar
                )
            )
            content_layout.addWidget(
                self._visualization_image_panel(
                    "<b><i>ρ<sub>e</sub></i></b> — "
                    "<span style='font-size:14px;'>electron density "
                    "(<i>e</i><sup>−</sup>·<i>cm</i><sup>−3</sup>)"
                    "</span>",
                    rhoe_pixmap,
                    rhoe_colorbar
                )
            )

            scroll_area.setWidget(content)
            dialog_layout.addWidget(scroll_area, 1)

            close_button = QtWidgets.QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            dialog_layout.addWidget(close_button)

            self.visualization_status_label.setText(
                "Visualization previews created successfully. "
                "The three results are open in a separate window."
            )

            self._visualization_dialog = dialog
            dialog.finished.connect(
                lambda _result: setattr(
                    self,
                    "_visualization_dialog",
                    None
                )
            )
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()

        except Exception as error:
            self.visualization_status_label.setText(
                f"Visualization preview failed: {error}"
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Geo_DECT — Visualization preview failed",
                str(error)
            )

    @staticmethod
    def _parse_dicom_multi_float(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            parts = value
        else:
            parts = str(value).replace(",", "\\").split("\\")
        output = []
        for part in parts:
            try:
                output.append(float(str(part).strip()))
            except (TypeError, ValueError):
                pass
        return output

    def _volume_spacing_xyz_m(self):
        """Infer native voxel spacing in meters from DICOM or TIFF metadata.

        A manually entered value in voxel_size_override_edit (micrometres)
        always wins over auto-detection, for both DICOM and TIFF -- this
        exists specifically so metadata parsing mistakes (missing tags,
        unrecognized units, wrong convention) can be corrected by the
        user directly rather than silently trusted. The override accepts
        one value (isotropic) or three values X, Y, Z (anisotropic).
        """
        override_text = self.voxel_size_override_edit.text().strip()
        if override_text:
            override_um = self._parse_voxel_size_override(override_text)
            if override_um is None:
                raise ValueError(
                    "Voxel size override not understood: '{}'. Enter "
                    "one positive value (isotropic) or three values "
                    "'X, Y, Z' (anisotropic), in µm.".format(override_text)
                )
            return tuple(v * 1.0e-6 for v in override_um)

        return self._auto_detect_spacing_xyz_m()

    @staticmethod
    def _parse_voxel_size_override(text):
        """Parse the voxel-size override field (µm).

        Accepts '50' (isotropic) or 'X, Y, Z' with commas, semicolons,
        spaces or 'x'/'×' as separators, e.g. '488.3, 488.3, 1250' or
        '488.3 x 488.3 x 1250'. Returns an (x, y, z) tuple in µm, or None
        if the text is not one or three finite positive numbers.
        """
        normalized = re.sub(r"[;,×xX*]", " ", str(text))
        parts = normalized.split()
        try:
            values = [float(part) for part in parts]
        except ValueError:
            return None
        if len(values) == 1:
            values = values * 3
        if len(values) != 3:
            return None
        if not all(np.isfinite(v) and v > 0 for v in values):
            return None
        return tuple(values)

    @classmethod
    def _format_spacing_um(cls, spacing_xyz_um):
        """Format X, Y, Z spacing (µm) for the override field: a single
        number when the voxel is isotropic, 'X, Y, Z' otherwise."""
        x_um, y_um, z_um = (float(v) for v in spacing_xyz_um)
        reference = max(abs(x_um), abs(y_um), abs(z_um), 1e-30)
        if (
            abs(x_um - y_um) / reference < 1e-6
            and abs(x_um - z_um) / reference < 1e-6
        ):
            return cls._format_number(x_um)
        return ", ".join(
            cls._format_number(v) for v in (x_um, y_um, z_um)
        )

    def _dicom_slice_spacing_from_positions_mm(self):
        """Slice spacing (mm) from ImagePositionPatient of two consecutive
        slices, projected onto the slice normal (ImageOrientationPatient).

        This is the true centre-to-centre distance between reconstructed
        slices. SliceThickness is NOT a spacing: with overlapping
        reconstructions (e.g. 1.25 mm thick slices every 0.625 mm) it
        overestimates Z. Returns None if the tags are missing.
        """
        if len(self.low_files) < 2:
            return None
        _raw0, by_name0, _tag0 = self._dicom_attributes(self.low_files[0])
        _raw1, by_name1, _tag1 = self._dicom_attributes(self.low_files[1])
        position0 = self._parse_dicom_multi_float(
            by_name0.get("ImagePositionPatient")
        )
        position1 = self._parse_dicom_multi_float(
            by_name1.get("ImagePositionPatient")
        )
        if len(position0) < 3 or len(position1) < 3:
            return None
        delta = np.asarray(position1[:3]) - np.asarray(position0[:3])

        orientation = self._parse_dicom_multi_float(
            by_name0.get("ImageOrientationPatient")
        )
        if len(orientation) >= 6:
            normal = np.cross(
                np.asarray(orientation[:3]), np.asarray(orientation[3:6])
            )
            norm = float(np.linalg.norm(normal))
            if norm > 0:
                spacing = abs(float(np.dot(delta, normal / norm)))
            else:
                spacing = float(np.linalg.norm(delta))
        else:
            spacing = float(np.linalg.norm(delta))

        if not np.isfinite(spacing) or spacing <= 0:
            return None
        return spacing

    def _auto_detect_spacing_xyz_m(self):
        """The actual DICOM/TIFF metadata parsing, without the manual
        override above -- kept separate so the "Check datasets" step can
        call this directly to pre-fill voxel_size_override_edit with a
        starting value for the user to confirm or correct."""
        if self.low_format == "dicom":
            _raw, by_name, _by_tag = self._dicom_attributes(
                self.low_files[0]
            )
            pixel_spacing = self._parse_dicom_multi_float(
                by_name.get("PixelSpacing")
            )
            if len(pixel_spacing) >= 2:
                y_spacing_mm = pixel_spacing[0]
                x_spacing_mm = pixel_spacing[1]
            else:
                x_spacing_mm = y_spacing_mm = 1.0

            # Z priority: real slice positions > SpacingBetweenSlices >
            # SliceThickness (the last one is a thickness, not a spacing,
            # and is only a fallback).
            try:
                z_spacing_mm = (
                    self._dicom_slice_spacing_from_positions_mm()
                )
            except Exception:
                z_spacing_mm = None
            if z_spacing_mm is None:
                z_spacing_mm = self._attribute_float(
                    by_name,
                    "SpacingBetweenSlices",
                    None
                )
            if z_spacing_mm is None:
                z_spacing_mm = self._attribute_float(
                    by_name,
                    "SliceThickness",
                    1.0
                )

            return (
                x_spacing_mm * 1.0e-3,
                y_spacing_mm * 1.0e-3,
                abs(z_spacing_mm) * 1.0e-3
            )

        # TIFF: try ImageJ spacing and resolution tags. Fall back to one
        # arbitrary unit per voxel only when no physical metadata exists.
        with tf.TiffFile(self.low_files[0]) as tif:
            page = tif.pages[0]
            imagej = tif.imagej_metadata or {}
            unit = str(imagej.get("unit", "")).lower()

            unit_scale = {
                "m": 1.0,
                "meter": 1.0,
                "metre": 1.0,
                "cm": 1.0e-2,
                "centimeter": 1.0e-2,
                "centimetre": 1.0e-2,
                "mm": 1.0e-3,
                "millimeter": 1.0e-3,
                "millimetre": 1.0e-3,
                "um": 1.0e-6,
                "µm": 1.0e-6,
                "micron": 1.0e-6,
                "nm": 1.0e-9,
            }.get(unit, 1.0)

            x_spacing = y_spacing = unit_scale
            try:
                x_resolution = page.tags["XResolution"].value
                y_resolution = page.tags["YResolution"].value
                x_ppu = float(x_resolution[0]) / float(x_resolution[1])
                y_ppu = float(y_resolution[0]) / float(y_resolution[1])
                if x_ppu > 0:
                    x_spacing = unit_scale / x_ppu
                if y_ppu > 0:
                    y_spacing = unit_scale / y_ppu
            except Exception:
                pass

            # Z spacing: ImageJ's own "spacing" tag, in the same unit as
            # above, when present. If the stack has no such metadata
            # (common for TIFF stacks that only carry XResolution/
            # YResolution), falling back to unit_scale alone silently
            # assumes "1 unrecognized-unit == 1 metre" -- for a 500+
            # slice micro-CT stack that turns into a physically absurd
            # multi-hundred-metre depth (e.g. 1000 mm per slice) while
            # X/Y stay correct, stretching the volume until the frontal
            # and sagittal views become blank. Reusing the ALREADY
            # correctly-derived in-plane spacing is a far safer default
            # than guessing "1 metre" out of thin air.
            has_explicit_z_spacing = (
                "spacing" in imagej and str(imagej.get("unit", "")).strip()
            )
            if has_explicit_z_spacing:
                z_value = imagej.get("spacing", 1.0)
                z_spacing = float(z_value) * unit_scale
            else:
                z_spacing = (x_spacing + y_spacing) / 2.0

            return (
                float(x_spacing),
                float(y_spacing),
                float(z_spacing)
            )

    @staticmethod
    def _running_positive_range(current, values):
        finite = np.asarray(values)
        finite = finite[
            np.isfinite(finite) & (finite > 0.0)
        ]
        if finite.size == 0:
            return current

        local_min = float(np.min(finite))
        local_max = float(np.max(finite))
        if current is None:
            return [local_min, local_max]

        current[0] = min(current[0], local_min)
        current[1] = max(current[1], local_max)
        return current

    def _build_and_publish_3d_volumes(self):
        """Calculate and publish three native Dragonfly volumetric objects.

        Calculations are first accumulated in ordinary NumPy arrays. Only
        after the whole stack has been validated are native Dragonfly Channels
        created through ``Channel().fromNDArray(...)``. Dragonfly 2025.1
        returns the initialized Channel from that call; retaining that returned
        object is essential.
        """
        # If a "Geo_DECT display range" dock from a PREVIOUS Build is still
        # open, _show_live_display_range_dock() below would just re-show it
        # as-is rather than recreate it -- so its combined scatterplot
        # widget would keep whatever settings (including rhoe_display_scale)
        # were correct/current back when IT was first created, even after
        # this fresh Build computes new, current ones. Closing it here forces
        # a clean rebuild against the settings this Build is about to
        # produce, the same cleanup _reset_analysis() already does.
        if self._display_range_dock is not None:
            try:
                self._display_range_dock.close()
                self._display_range_dock.deleteLater()
            except Exception:
                pass
            self._display_range_dock = None
            self._display_range_controls = {}

        progress = None

        try:
            from . import geodect_3d as geodect3d

            estimate_output_bytes = geodect3d.estimate_output_bytes
            format_bytes = geodect3d.format_bytes
            create_native_channel = geodect3d.create_native_channel
            encode_geodect_slice = geodect3d.encode_geodect_slice
            create_geodect_lut = geodect3d.create_geodect_lut
            create_context_scalar_lut = geodect3d.create_context_scalar_lut
            finish_channel = geodect3d.finish_channel

            if self.combined_bins_122_radio.isChecked():
                selected_combined_bins = 122
            elif self.combined_bins_100_radio.isChecked():
                selected_combined_bins = 100
            else:
                selected_combined_bins = 70
            geodect3d.set_geodect_bins(selected_combined_bins)

            if not self.low_files or not self.high_files:
                raise ValueError(
                    "Check the LOW and HIGH datasets first."
                )
            if len(self.low_files) != len(self.high_files):
                raise ValueError(
                    "LOW and HIGH stacks contain different slice counts."
                )

            self._validate_manual_physical_ranges()

            zeff_degree = int(self.zeff_degree_combo.currentText())
            rhoe_degree = int(self.rhoe_degree_combo.currentText())
            zeff_coefficients = self._read_coefficients(
                self.zeff_coeff_edits,
                zeff_degree,
                "Zeff"
            )
            rhoe_coefficients = self._read_coefficients(
                self.rhoe_coeff_edits,
                rhoe_degree,
                "rho_e"
            )
            ratio_min, ratio_max = self._read_ratio_bounds()
            denoise_method, denoise_strength = (
                self._read_denoise_settings()
            )
            _html, numerator_mode = self._selected_rhoe_numerator_html()

            if not hasattr(self, "test_zeff"):
                raise ValueError(
                    "Run the single-slice calculation once so that the "
                    "display ranges can be initialized."
                )

            rhoe_reference, _symbol, _unit, _description = (
                self._visualization_rhoe_data()
            )
            zeff_minimum, zeff_maximum = self._read_visualization_range(
                self.visualization_zeff_min_edit,
                self.visualization_zeff_max_edit,
                self.test_zeff,
                "Zeff"
            )
            rhoe_minimum, rhoe_maximum = self._read_visualization_range(
                self.visualization_rhoe_min_edit,
                self.visualization_rhoe_max_edit,
                rhoe_reference,
                "rho_e"
            )

            if self.low_format == "tiff":
                first_low = tf.imread(self.low_files[0])
                first_high = tf.imread(self.high_files[0])
            else:
                first_low, _low_hu, _low_mu, _low_info = self._read_dicom(
                    self.low_files[0]
                )
                first_high, _high_hu, _high_mu, _high_info = self._read_dicom(
                    self.high_files[0]
                )

            if first_low.ndim != 2 or first_high.ndim != 2:
                raise ValueError(
                    "Each LOW/HIGH source image must be a two-dimensional slice."
                )
            if first_low.shape != first_high.shape:
                raise ValueError(
                    "LOW and HIGH slice dimensions do not match."
                )

            shape_zyx = (
                len(self.low_files),
                int(first_low.shape[0]),
                int(first_low.shape[1])
            )
            native_output_bytes = estimate_output_bytes(shape_zyx)
            # During conversion both the local NumPy volumes and the native
            # Dragonfly copies coexist temporarily.
            estimated_peak_bytes = native_output_bytes * 2

            answer = QtWidgets.QMessageBox.question(
                self,
                "Geo_DECT — Build native 3D volumes",
                "Geo_DECT will calculate three NumPy volumes and then create "
                "three native Dragonfly objects.\n\n"
                f"Volume shape: {shape_zyx[2]} × {shape_zyx[1]} × "
                f"{shape_zyx[0]} voxels\n"
                f"Native output memory: {format_bytes(native_output_bytes)}\n"
                f"Estimated peak during transfer: "
                f"{format_bytes(estimated_peak_bytes)}\n\n"
                "The computation is performed slice by slice. Continue?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

            spacing_xyz_m = self._volume_spacing_xyz_m()

            self.create_3d_volumes_button.setEnabled(False)
            self.volume_progress_bar.setValue(0)
            self.volume_status_label.setText(
                "Allocating temporary NumPy volumes..."
            )
            QtWidgets.QApplication.processEvents()

            # Zero is the explicit background value. Keeping NaN out of the
            # native float channels makes Dragonfly statistics and window/level
            # initialization deterministic.
            zeff_volume = np.zeros(shape_zyx, dtype=np.float32)
            rhoe_volume = np.zeros(shape_zyx, dtype=np.float32)
            combined_volume = np.zeros(shape_zyx, dtype=np.uint16)

            zeff_range = None
            rhoe_range = None
            # Decided on the first slice that has valid rho_e values and
            # then kept for the whole stack (see
            # _rhoe_needs_water_conversion).
            rhoe_convert_decision = None
            nonphysical_zeff = 0
            nonphysical_rhoe = 0

            progress = QtWidgets.QProgressDialog(
                "Calculating Geo_DECT 3D volumes...",
                "Cancel",
                0,
                len(self.low_files),
                self
            )
            progress.setWindowModality(
                QtCore.Qt.WindowModality.WindowModal
            )
            progress.setMinimumDuration(0)

            for z_index, (low_path, high_path) in enumerate(
                zip(self.low_files, self.high_files)
            ):
                if progress.wasCanceled():
                    raise RuntimeError(
                        "3D calculation cancelled by the user."
                    )

                if self.low_format == "tiff":
                    low_gray = tf.imread(low_path)
                    high_gray = tf.imread(high_path)
                    low_hu = None
                    high_hu = None
                    mu_low = compute_mu(low_gray, self.low_calibration)
                    mu_high = compute_mu(high_gray, self.high_calibration)
                else:
                    low_gray, low_hu, mu_low, _ = self._read_dicom(low_path)
                    high_gray, high_hu, mu_high, _ = self._read_dicom(high_path)

                if low_gray.shape != first_low.shape:
                    raise ValueError(
                        f"LOW slice {z_index} has inconsistent dimensions."
                    )
                if high_gray.shape != first_high.shape:
                    raise ValueError(
                        f"HIGH slice {z_index} has inconsistent dimensions."
                    )

                valid_mask, _threshold = self._background_mask(
                    high_gray,
                    high_hu
                )
                if valid_mask is None:
                    retained_mask = np.ones(
                        high_gray.shape,
                        dtype=bool
                    )
                else:
                    retained_mask = np.asarray(valid_mask, dtype=bool)

                mu_low, mu_high = self._denoise_mu_pair(
                    mu_low,
                    mu_high,
                    retained_mask,
                    denoise_method,
                    denoise_strength,
                )

                zeff = compute_zeff(
                    mu_low,
                    mu_high,
                    zeff_coefficients,
                    valid_mask=retained_mask,
                    ratio_min=ratio_min,
                    ratio_max=ratio_max
                )
                rhoe_relative = compute_rhoe(
                    mu_low,
                    mu_high,
                    zeff,
                    rhoe_coefficients,
                    numerator_mode=numerator_mode,
                    valid_mask=retained_mask
                )

                if rhoe_convert_decision is None:
                    rhoe_convert_decision = (
                        self._rhoe_needs_water_conversion(rhoe_relative)
                    )
                if rhoe_convert_decision:
                    rhoe = relative_to_absolute_rhoe(
                        rhoe_relative,
                        WATER_ELECTRON_DENSITY_CM3
                    )
                else:
                    rhoe = rhoe_relative

                zeff = np.asarray(zeff)
                rhoe = np.asarray(rhoe)
                physical = (
                    retained_mask
                    & np.isfinite(zeff)
                    & np.isfinite(rhoe)
                    & (zeff > 0.0)
                    & (rhoe > 0.0)
                )

                nonphysical_zeff += int(
                    np.count_nonzero(
                        retained_mask & np.isfinite(zeff) & (zeff <= 0.0)
                    )
                )
                nonphysical_rhoe += int(
                    np.count_nonzero(
                        retained_mask & np.isfinite(rhoe) & (rhoe <= 0.0)
                    )
                )

                zeff_slice = zeff_volume[z_index]
                rhoe_slice = rhoe_volume[z_index]
                # Preserve the physical Z_eff values directly.  The old
                # implementation multiplied by 100 and stored uint16 values,
                # which produced a native Dragonfly range such as 0..2887
                # instead of the intended physical range around 10..18.
                zeff_slice[physical] = zeff[physical].astype(
                    np.float32,
                    copy=False
                )
                rhoe_slice[physical] = rhoe[physical].astype(
                    np.float32,
                    copy=False
                )
                zeff_range = self._running_positive_range(
                    zeff_range,
                    zeff_slice
                )
                rhoe_range = self._running_positive_range(
                    rhoe_range,
                    rhoe_slice
                )

                progress.setValue(z_index + 1)
                self.volume_progress_bar.setValue(
                    int(100 * (z_index + 1) / len(self.low_files))
                )
                self.volume_status_label.setText(
                    f"Processing slice {z_index + 1} of "
                    f"{len(self.low_files)}..."
                )
                QtWidgets.QApplication.processEvents()

            progress.close()
            progress = None

            if zeff_range is None or rhoe_range is None:
                raise ValueError(
                    "No positive physical voxels were produced in the stack."
                )

            # The 3D output must use the automatic range of THIS stack,
            # not the range measured on the earlier test slice.  These are
            # the same physical arrays used to build the native volumes.
            zeff_stack_minimum, zeff_stack_maximum = map(float, zeff_range)
            rhoe_minimum, rhoe_maximum = map(float, rhoe_range)

            # Zeff often contains a small number of extreme positive outliers.
            # Using the literal full range as the initial window compresses the
            # useful anatomy into a nearly uniform image. Keep the true stack
            # limits as slider bounds, but initialize the visible Zeff interval
            # from robust whole-stack percentiles. No voxel data are changed.
            zeff_positive = zeff_volume[
                np.isfinite(zeff_volume) & (zeff_volume > 0.0)
            ]
            if zeff_positive.size >= 32:
                zeff_minimum = float(np.percentile(zeff_positive, 1.0))
                zeff_maximum = float(np.percentile(zeff_positive, 99.0))
                if zeff_maximum <= zeff_minimum:
                    zeff_minimum = zeff_stack_minimum
                    zeff_maximum = zeff_stack_maximum
            else:
                zeff_minimum = zeff_stack_minimum
                zeff_maximum = zeff_stack_maximum

            # Keep every positive finite voxel included by retained_mask,
            # including genuine outliers.  For Dragonfly display only,
            # absolute electron density is expressed in units of 10^23
            # e^- cm^-3.  This avoids unreadable 24-digit legend labels while
            # preserving the physical calculation and the CIE Lab crossplot.
            #
            # Which calibrations need this scaling is NOT determined by
            # file format (TIFF vs DICOM) -- a TIFF dataset calibrated
            # with coefficients that yield ABSOLUTE rho_e (~1e23, same
            # order as any DICOM-derived one) needs it just as much as
            # any DICOM does. With the format-based rule the scale
            # silently stayed at 1.0 for such TIFFs, so the display
            # range showed raw values with no reduction, and every
            # colorbar tick became a very long unscaled number -- with
            # only ~300px of width for 4 ticks, those long strings ran
            # into each other into a wall of overlapping digits.
            # Basing the decision on the actual magnitude of the
            # computed rho_e instead of the file format fixes this for
            # any dataset/format/calibration combination.
            rhoe_display_scale = (
                1.0e23 if abs(rhoe_maximum) >= 100.0 else 1.0
            )
            rhoe_display_minimum = rhoe_minimum / rhoe_display_scale
            rhoe_display_maximum = rhoe_maximum / rhoe_display_scale

            self.visualization_zeff_min_edit.setText(
                self._format_range_input(zeff_minimum)
            )
            self.visualization_zeff_max_edit.setText(
                self._format_range_input(zeff_maximum)
            )
            self.visualization_rhoe_min_edit.setText(
                self._format_range_input(rhoe_minimum, scientific=True)
            )
            self.visualization_rhoe_max_edit.setText(
                self._format_range_input(rhoe_maximum, scientific=True)
            )

            # Encode the two-dimensional CIE Lab crossplot only after the
            # complete-stack limits are known.  Encoding slice-by-slice with
            # the test-slice limits produced a different result from preview.
            for z_index in range(shape_zyx[0]):
                zeff_slice = zeff_volume[z_index]
                rhoe_slice = rhoe_volume[z_index]
                physical = (zeff_slice > 0.0) & (rhoe_slice > 0.0)
                combined_volume[z_index] = encode_geodect_slice(
                    zeff_slice,
                    rhoe_slice,
                    physical,
                    zeff_stack_minimum,
                    zeff_stack_maximum,
                    rhoe_minimum,
                    rhoe_maximum,
                )

            # The Combined codes above were generated from the full physical
            # rho_e values.  Only now convert the standalone rho_e channel to
            # display units.  Its numeric legend will therefore read, e.g.,
            # 0.93...8.91 with the explicit multiplier x10^23 in the title.
            if rhoe_display_scale != 1.0:
                rhoe_volume /= np.float32(rhoe_display_scale)

            # Keep independent copies of the true physical values, matching
            # exactly what goes into the Z_eff/rho_e channels below. The
            # "Apply range" button (per-property discrete/context display,
            # added 2026-08) overwrites the channel's own data with integer
            # display codes -- Probe and the combined-volume "Apply" button
            # both need the real physical numbers, not those codes, so they
            # read from these stored copies instead of the channel directly.
            self.geodect_3d_raw_arrays = {
                "zeff": zeff_volume.copy(),
                "rhoe": rhoe_volume.copy(),
            }

            self.volume_status_label.setText(
                "Creating and validating native Dragonfly channels..."
            )
            QtWidgets.QApplication.processEvents()

            # create_native_channel now accepts the COMPLETE NumPy volume and
            # retains the initialized Channel returned by fromNDArray().
            #
            # TEMPORARY diagnostic (2026-08): a user hit Dragonfly's
            # "Data was not initialized... dtype not supported" error on
            # a specific dataset, even though a console test confirmed
            # float32 itself works fine on a trivial array -- so the
            # real cause is something about THIS array's actual content,
            # not the dtype in general. If channel creation fails here,
            # write out the real array's characteristics (shape, min/max,
            # NaN/Inf presence, contiguity) before re-raising, so the
            # true cause can be pinned down instead of guessed at again.
            # Shared by both Z_eff and rho_e below -- the same failure
            # has now been seen on both channels on this dataset, even
            # after clipping obviously-implausible values in
            # geodect_core.py, so this needs to keep catching whichever
            # channel fails next rather than being Z_eff-only.
            def _memory_report_lines():
                """Best-effort system RAM snapshot for the diagnostic.

                psutil is not guaranteed to be present in Dragonfly's
                internal Python environment (same situation already
                handled for matplotlib/scipy elsewhere in this
                plugin) -- if it's missing, this simply contributes
                no lines instead of blocking the diagnostic.
                """
                try:
                    import psutil

                    virtual_memory = psutil.virtual_memory()
                    process = psutil.Process(os.getpid())
                    process_bytes = process.memory_info().rss
                    return [
                        "system RAM total: {}".format(
                            format_bytes(virtual_memory.total)
                        ),
                        "system RAM available: {}".format(
                            format_bytes(virtual_memory.available)
                        ),
                        "this process RSS: {}".format(
                            format_bytes(process_bytes)
                        ),
                    ]
                except Exception:
                    return []

            def create_channel_with_diagnostic(
                volume, dtype, title, display_name
            ):
                try:
                    return create_native_channel(
                        volume, dtype, title, spacing_xyz_m
                    )
                except Exception as channel_exc:
                    try:
                        diagnostic_path = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "geodect_channel_diagnostic.txt",
                        )
                        finite_mask = np.isfinite(volume)
                        report_lines = [
                            "Geo_DECT channel-creation diagnostic ({})"
                            .format(display_name),
                            "Exception: {}: {}".format(
                                type(channel_exc).__name__, channel_exc
                            ),
                            "shape: {}".format(volume.shape),
                            "dtype: {}".format(volume.dtype),
                            "C-contiguous: {}".format(
                                volume.flags["C_CONTIGUOUS"]
                            ),
                            "size (voxels): {}".format(volume.size),
                            "nbytes: {}".format(volume.nbytes),
                            "all finite: {}".format(
                                bool(np.all(finite_mask))
                            ),
                            "NaN count: {}".format(
                                int(np.count_nonzero(np.isnan(volume)))
                            ),
                            "+Inf count: {}".format(
                                int(np.count_nonzero(np.isposinf(volume)))
                            ),
                            "-Inf count: {}".format(
                                int(np.count_nonzero(np.isneginf(volume)))
                            ),
                        ]
                        if np.any(finite_mask):
                            finite_values = volume[finite_mask]
                            report_lines.append(
                                "finite min/max: {} / {}".format(
                                    float(np.min(finite_values)),
                                    float(np.max(finite_values)),
                                )
                            )
                            # A short histogram of magnitudes tells us
                            # whether it's a handful of extreme outliers
                            # or the bulk of the data that's out of
                            # Dragonfly's comfort zone.
                            abs_finite = np.abs(finite_values)
                            for threshold in (
                                1e3, 1e6, 1e9, 1e12, 1e15, 1e18, 1e21, 1e24,
                            ):
                                count_above = int(
                                    np.count_nonzero(abs_finite > threshold)
                                )
                                if count_above:
                                    report_lines.append(
                                        "voxels with |value| > {:.0e}: {}"
                                        .format(threshold, count_above)
                                    )
                        report_lines.extend(_memory_report_lines())
                        with open(
                            diagnostic_path, "w", encoding="utf-8"
                        ) as handle:
                            handle.write("\n".join(report_lines))
                        QtWidgets.QMessageBox.critical(
                            self,
                            "Geo_DECT — channel creation failed",
                            "Failed to create the {} channel.\n\n"
                            .format(display_name)
                            + "\n".join(report_lines)
                            + "\n\nAlso saved to:\n{}".format(
                                diagnostic_path
                            ),
                        )
                    except Exception:
                        pass
                    raise

            # Each of the three volumes (Z_eff, rho_e, combined) plus the
            # two independent copies already stashed in
            # geodect_3d_raw_arrays can add up to a very large peak
            # memory footprint on big datasets. Rather than hold all
            # three big local arrays alive until the very end (as
            # before), release each one as soon as its own channel has
            # been created and validated -- so memory pressure goes
            # DOWN one step at a time instead of only accumulating,
            # giving the later channels (rho_e, then combined) more
            # headroom on datasets that are large enough to matter.
            # geodect_3d_raw_arrays keeps independent copies already,
            # so nothing downstream (Probe, combined "Apply") is lost.
            zeff_channel = create_channel_with_diagnostic(
                zeff_volume,
                np.float32,
                "Geo_DECT — Z_eff (effective atomic number)",
                "Z_eff",
            )
            del zeff_volume
            gc.collect()

            rhoe_title = (
                "Geo_DECT — rho_e (×10^23 e^-·cm^-3)"
                if rhoe_display_scale != 1.0
                else "Geo_DECT — rho_e (relative electron density)"
            )
            rhoe_channel = create_channel_with_diagnostic(
                rhoe_volume,
                np.float32,
                rhoe_title,
                "rho_e",
            )
            del rhoe_volume
            gc.collect()

            # Now also diagnosed the same way as Z_eff/rho_e above: this
            # channel was seen crashing with the same generic Dragonfly
            # error even on a dataset already small enough for the
            # other two channels to succeed -- most likely explained by
            # cumulative memory pressure (see the del/gc.collect() calls
            # above and below) rather than a per-array content problem,
            # but keep the full diagnostic here too in case it recurs.
            combined_channel = create_channel_with_diagnostic(
                combined_volume,
                np.uint16,
                "Geo_DECT — combined Z_eff / rho_e",
                "Combined",
            )
            del combined_volume
            gc.collect()

            # Native min/max include the explicit zero background and are the
            # correct limits for mapping a continuous LUT onto each channel.
            zeff_data_minimum = float(zeff_channel.getMinimumValue())
            zeff_data_maximum = float(zeff_channel.getMaximumValue())
            rhoe_data_minimum = float(rhoe_channel.getMinimumValue())
            rhoe_data_maximum = float(rhoe_channel.getMaximumValue())

            self.volume_status_label.setText(
                "Publishing channels and creating LUTs..."
            )
            QtWidgets.QApplication.processEvents()

            finish_channel(
                zeff_channel,
                zeff_minimum,
                zeff_maximum,
            )
            finish_channel(
                rhoe_channel,
                rhoe_display_minimum,
                rhoe_display_maximum,
            )
            finish_channel(
                combined_channel,
                0,
                geodect3d.GEODECT_MAX_CODE
            )

            # Keep the native model objects referenced while their visual
            # representations, clip boxes, legends and LUTs are created.
            self.geodect_3d_channels = {
                "zeff": zeff_channel,
                "rhoe": rhoe_channel,
                "combined": combined_channel,
            }
            self.geodect_3d_display_settings = {
                "zeff_minimum": float(zeff_minimum),
                "zeff_maximum": float(zeff_maximum),
                "zeff_full_minimum": float(zeff_stack_minimum),
                "zeff_full_maximum": float(zeff_stack_maximum),
                # Physical limits remain available for reporting and for
                # the two-dimensional CIE Lab mapping.  The LUT/window values
                # use the scaled display coordinates of the native channel.
                "rhoe_physical_minimum": float(rhoe_minimum),
                "rhoe_physical_maximum": float(rhoe_maximum),
                "rhoe_display_scale": float(rhoe_display_scale),
                "rhoe_minimum": float(rhoe_display_minimum),
                "rhoe_maximum": float(rhoe_display_maximum),
                "rhoe_full_minimum": float(rhoe_display_minimum),
                "rhoe_full_maximum": float(rhoe_display_maximum),
            }

            # Tag the range onto the channels themselves so it survives
            # a Dragonfly project save/reload (see
            # _reconnect_existing_geodect_channels below -- this Python
            # dict alone does not survive it).
            geodect3d.persist_display_settings(
                zeff_channel, self.geodect_3d_display_settings, "continuous"
            )
            geodect3d.persist_display_settings(
                rhoe_channel, self.geodect_3d_display_settings, "continuous"
            )
            geodect3d.persist_display_settings(combined_channel, {}, "discrete")

            # Create the VisualChannels directly and retain the returned
            # objects. Channel.publish() only publishes the model object; it
            # does not guarantee that Channel.getVisual() will resolve the
            # visual created for the current layout.
            from OrsHelpers.volumehelper import VolumeHelper
            from OrsHelpers.layoutHelper import LayoutHelper
            from OrsLibraries.workingcontext import WorkingContext
            from ORSServiceClass.preferencesapplier.preferencesApplyManager import (
                PreferencesApplyManager,
            )
            from OrsHelpers.layoutpropertieshelper import LayoutPropertiesHelper

            current_view = WorkingContext.getCurrentView(None)
            if current_view is None:
                raise RuntimeError(
                    "Dragonfly has no current view in which to create the "
                    "Geo_DECT volumes."
                )

            scene_layout = LayoutHelper.getSceneLayoutOfView(current_view)
            if scene_layout is None:
                raise RuntimeError(
                    "Dragonfly could not resolve the scene layout of the "
                    "current view."
                )

            visuals = {}
            clip_boxes = {}
            legends = {}
            visual_errors = {}

            # Create Z_eff last so Dragonfly selects/renders it immediately
            # instead of requiring a click on another scalar volume first.
            for key in ("rhoe", "combined", "zeff"):
                channel = self.geodect_3d_channels[key]
                try:
                    visual, clip_box, legend = (
                        VolumeHelper.createVolumeForChannel(
                            channel,
                            layout=scene_layout,
                        )
                    )
                    if visual is None:
                        raise RuntimeError(
                            "VolumeHelper.createVolumeForChannel returned "
                            "no VisualChannel."
                        )

                    visuals[key] = visual
                    clip_boxes[key] = clip_box
                    legends[key] = legend

                    # A one-dimensional numeric legend is misleading for the
                    # combined channel because each code represents a point in
                    # the two-dimensional (Z_eff, rho_e) CIE Lab crossplot.
                    if key == "combined" and legend is not None:
                        for method_name in ("setIsVisible", "setVisible"):
                            method = getattr(legend, method_name, None)
                            if callable(method):
                                try:
                                    method(False)
                                    break
                                except Exception:
                                    pass

                    try:
                        PreferencesApplyManager.applyPreferencesOn(visual)
                    except Exception:
                        # Preferences are cosmetic; a failure here must not
                        # discard an otherwise valid visual representation.
                        pass

                    # Use the selected physical display interval instead of
                    # Dragonfly's automatic full native range.  The combined
                    # channel remains an indexed CIE Lab crossplot volume.
                    if key == "zeff":
                        display_min = float(zeff_minimum)
                        display_max = float(zeff_maximum)
                    elif key == "rhoe":
                        display_min = float(rhoe_display_minimum)
                        display_max = float(rhoe_display_maximum)
                    else:
                        display_min = 0.0
                        display_max = float(geodect3d.GEODECT_MAX_CODE)

                    display_width = display_max - display_min
                    display_center = (display_max + display_min) / 2.0
                    visual.set2DWindowLevelValuesForAllViews(
                        display_width, display_center
                    )
                    visual.set3DWindowLevelValuesForAllViews(
                        display_width, display_center
                    )
                    # Start all three volumes hidden (eye icon crossed out).
                    # The person turns on whichever visualization(s) they
                    # want, one at a time, instead of seeing all three
                    # stacked together right after creation.
                    for display in scene_layout.getAllChildViews():
                        visual.setIsVisibleForView(display, False)
                    visual.triggerClassEvent("OrsObjectPublishedEvent")
                    visual.setPropertyDirty()
                    visual.refreshAllParentViews()

                except Exception as exc:
                    visual_errors[key] = (
                        f"{type(exc).__name__}: {exc}"
                    )

            if visual_errors or len(visuals) != 3:
                details = "; ".join(
                    f"{key}={message}"
                    for key, message in sorted(visual_errors.items())
                )
                raise RuntimeError(
                    "Dragonfly could not create all three Geo_DECT "
                    "VisualChannels directly through VolumeHelper. "
                    + details
                )

            self.geodect_3d_visuals = visuals
            self.geodect_3d_clip_boxes = clip_boxes
            self.geodect_3d_legends = legends
            self.geodect_3d_scalar_collections = {}
            self.geodect_3d_scene_layout = scene_layout
            self.geodect_3d_current_view = current_view
            self.geodect_3d_visual_initialization = {
                "attempt": 1,
                "completed": True,
                "errors": {},
                "mode": "direct_volumehelper",
            }

            try:
                LayoutPropertiesHelper.updateSceneProperties(
                    scene_layout,
                    datasetIsVisible=True,
                )
            except Exception:
                pass

            # zeff_volume / rhoe_volume / combined_volume were already
            # released right after each channel was created and
            # validated above (freeing memory progressively instead of
            # only at the very end).

            self.geodect_3d_luts = {}
            lut_results = self._create_and_apply_geodect_3d_luts()
            self.geodect_3d_visual_initialization["lut_results"] = dict(
                lut_results
            )

            for key, visual in visuals.items():
                try:
                    if key == "zeff":
                        display_min = float(zeff_minimum)
                        display_max = float(zeff_maximum)
                    elif key == "rhoe":
                        display_min = float(rhoe_display_minimum)
                        display_max = float(rhoe_display_maximum)
                    else:
                        display_min = 0.0
                        display_max = float(geodect3d.GEODECT_MAX_CODE)

                    display_width = display_max - display_min
                    display_center = (display_max + display_min) / 2.0
                    visual.set2DWindowLevelValuesForAllViews(
                        display_width, display_center
                    )
                    visual.set3DWindowLevelValuesForAllViews(
                        display_width, display_center
                    )
                    visual.setPropertyDirty()
                    visual.refreshAllParentViews()
                except Exception:
                    pass
            try:
                current_view.refresh()
            except Exception:
                pass

            # Give Dragonfly one event cycle to finish registering all three
            # visuals, then refresh. All three volumes start hidden by
            # design (see above); this only keeps their internal state
            # fresh, it does not force any of them visible.
            def _refresh_initial_zeff():
                try:
                    visual = self.geodect_3d_visuals.get("zeff")
                    if visual is not None:
                        visual.setPropertyDirty()
                        visual.refreshAllParentViewsIfVisible()
                        visual.refreshAllParentViews()
                    LayoutHelper.refreshAllViewsInSceneLayoutOfView(
                        self.geodect_3d_current_view
                    )
                except Exception:
                    pass
            # First refresh synchronously, then repeat after Dragonfly has
            # completed its internal layout cloning.
            _refresh_initial_zeff()
            QtCore.QTimer.singleShot(0, _refresh_initial_zeff)
            QtCore.QTimer.singleShot(120, _refresh_initial_zeff)
            QtCore.QTimer.singleShot(350, _refresh_initial_zeff)

            # Runs strictly after every other refresh/window-level call
            # above (see _apply_confirmed_3d_coloring docstring for why
            # the ordering matters).
            # 600ms was confirmed (2026-08) to be too early -- Dragonfly is
            # not always ready yet, and the coloring silently has no visual
            # effect even though no exception is raised. 3s gives Dragonfly
            # more time to finish its own internal setup after volume
            # creation. The "Force 3D coloring" button remains as a manual
            # fallback if this is still too early on a slower machine or a
            # larger dataset.
            QtCore.QTimer.singleShot(3000, self._apply_confirmed_3d_coloring)

            self._show_live_display_range_dock()

            warning_text = ""
            if nonphysical_zeff or nonphysical_rhoe:
                warning_text = (
                    f" Non-physical retained voxels excluded: "
                    f"Z_eff={nonphysical_zeff:,}, "
                    f"rho_e={nonphysical_rhoe:,}."
                )

            self.volume_status_label.setText(
                "Three native Dragonfly volumes, VisualChannels and "
                "preview-matched color LUTs are ready in the current layout."
                + warning_text
            )

            QtWidgets.QMessageBox.information(
                self,
                "Geo_DECT — 3D volumes ready",
                "Three colored native volumes are now available in "
                "Dragonfly's object panel and current layout:\n\n"
                "• Geo_DECT — combined Z_eff / rho_e\n"
                "• Geo_DECT — Z_eff\n"
                "• Geo_DECT — rho_e\n\n"
                "They are native channels, so Dragonfly tools such as "
                "zoom, clipping/cropping, measuring and the Probe tool "
                "remain available."
            )

        except Exception as error:
            details = traceback.format_exc()
            self.geodect_3d_channels = {}
            self.volume_status_label.setText(
                f"3D volume generation failed: {error}"
            )
            message = QtWidgets.QMessageBox(self)
            message.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            message.setWindowTitle("Geo_DECT — 3D diagnostic error")
            message.setText(str(error))
            message.setInformativeText(
                "The 2D plugin remains available. Open 'Show Details', "
                "copy the complete traceback and send it for the next fix."
            )
            message.setDetailedText(details)
            message.exec()
        finally:
            if progress is not None:
                progress.close()
            self.create_3d_volumes_button.setEnabled(True)

    def _test_selected_tiff_pair(self):
        """Run the complete calculation on one selected image pair."""
        self.slice_test_results.clear()

        try:
            slice_index = self._selected_test_slice_index()
            self._validate_manual_physical_ranges()

            zeff_degree = int(
                self.zeff_degree_combo.currentText()
            )
            rhoe_degree = int(
                self.rhoe_degree_combo.currentText()
            )

            zeff_coefficients = self._read_coefficients(
                self.zeff_coeff_edits,
                zeff_degree,
                "Zeff"
            )
            rhoe_coefficients = self._read_coefficients(
                self.rhoe_coeff_edits,
                rhoe_degree,
                "rho_e"
            )
            ratio_min, ratio_max = self._read_ratio_bounds()
            denoise_method, denoise_strength = (
                self._read_denoise_settings()
            )

            _, numerator_mode = (
                self._selected_rhoe_numerator_html()
            )

            low_path = self.low_files[slice_index]
            high_path = self.high_files[slice_index]

            if self.low_format == "tiff":
                low_gray = tf.imread(low_path)
                high_gray = tf.imread(high_path)
                low_hu = None
                high_hu = None

                mu_low = compute_mu(
                    low_gray,
                    self.low_calibration
                )
                mu_high = compute_mu(
                    high_gray,
                    self.high_calibration
                )
            else:
                low_gray, low_hu, mu_low, _ = (
                    self._read_dicom(low_path)
                )
                high_gray, high_hu, mu_high, _ = (
                    self._read_dicom(high_path)
                )

            if low_gray.shape != high_gray.shape:
                raise ValueError(
                    "The selected LOW and HIGH images have different dimensions."
                )

            valid_mask, threshold = self._background_mask(
                high_gray,
                high_hu
            )

            mu_low, mu_high = self._denoise_mu_pair(
                mu_low,
                mu_high,
                valid_mask,
                denoise_method,
                denoise_strength,
            )

            zeff = compute_zeff(
                mu_low,
                mu_high,
                zeff_coefficients,
                valid_mask=valid_mask,
                ratio_min=ratio_min,
                ratio_max=ratio_max
            )

            rhoe_calibrated = compute_rhoe(
                mu_low,
                mu_high,
                zeff,
                rhoe_coefficients,
                numerator_mode=numerator_mode,
                valid_mask=valid_mask
            )

            report = [
                "SELECTED REAL IMAGE PAIR TEST",
                "",
                f"Format: {self.low_format.upper()}",
                f"Slice index: {slice_index}",
                f"LOW file: {os.path.basename(low_path)}",
                f"HIGH file: {os.path.basename(high_path)}",
                f"rho_e numerator mode: {numerator_mode}",
                "Noise reduction before ratio: {}".format(
                    self._describe_denoise(
                        denoise_method, denoise_strength
                    )
                ),
            ]

            if threshold is not None:
                threshold_unit = (
                    "HU"
                    if self.low_format == "dicom"
                    else "gray level"
                )
                report.append(
                    f"Background threshold: {threshold:g} {threshold_unit}"
                )

            report.append("")

            if self.low_format == "dicom":
                report.extend(
                    self._array_statistics("HU_LOW", low_hu)
                )
                report.append("")
                report.extend(
                    self._array_statistics("HU_HIGH", high_hu)
                )
                report.append("")

            report.extend(
                self._array_statistics("mu_LOW", mu_low)
            )
            report.append("")
            report.extend(
                self._array_statistics("mu_HIGH", mu_high)
            )
            report.append("")
            report.extend(
                self._array_statistics("Zeff", zeff)
            )
            report.append("")

            rhoe_converted = bool(
                self._rhoe_needs_water_conversion(rhoe_calibrated)
            )
            if rhoe_converted:
                report.extend(
                    self._array_statistics(
                        "rho_e / rho_e,water",
                        rhoe_calibrated
                    )
                )
                report.append("")

                rhoe_absolute = relative_to_absolute_rhoe(
                    rhoe_calibrated,
                    WATER_ELECTRON_DENSITY_CM3
                )
                report.extend(
                    self._array_statistics(
                        "rho_e absolute (electrons/cm^3)",
                        rhoe_absolute
                    )
                )
            else:
                rhoe_absolute = rhoe_calibrated
                report.extend(
                    self._array_statistics(
                        "rho_e",
                        rhoe_calibrated
                    )
                )

            self.slice_test_results.setPlainText(
                "\n".join(report)
            )

            self.test_slice_index = slice_index
            self.test_mu_low = mu_low
            self.test_mu_high = mu_high
            self.test_zeff = zeff
            self.test_rhoe = rhoe_calibrated
            self.test_rhoe_absolute = rhoe_absolute
            self.test_rhoe_converted = rhoe_converted
            self.test_valid_mask = valid_mask

            # Detect a positive physical range automatically. The user can
            # still edit the four fields to narrow the visualization.
            self._set_automatic_visualization_ranges()

        except Exception as error:
            self.slice_test_results.setPlainText(
                "Selected image-pair calculation failed:\n"
                f"{error}"
            )
            
 
        
            
    def _save_test_slice_tiffs(self):
        """Save the last single-slice test results as 32-bit TIFFs.

        Intended for independent checks outside Geo_DECT (ImageJ/Fiji):
        the same ROI can be applied to Z_eff, rho_e and the raw
        mu_LOW/mu_HIGH ratio. Voxels excluded by the background mask or
        by the plausible-range filters are stored as NaN, which ImageJ
        ignores in its statistics and histograms.
        """
        if not hasattr(self, "test_zeff"):
            QtWidgets.QMessageBox.information(
                self,
                "Geo_DECT \u2014 Save test slice",
                "Run the single-slice calculation first.",
            )
            return

        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder for the test-slice maps"
        )
        if not folder:
            return

        index = int(getattr(self, "test_slice_index", 0))
        mu_low = np.asarray(self.test_mu_low, dtype=np.float32)
        mu_high = np.asarray(self.test_mu_high, dtype=np.float32)

        # Raw ratio, WITHOUT the plausible-range filter: shows what the
        # calibration polynomial actually receives.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                mu_high > 0, mu_low / mu_high, np.nan
            ).astype(np.float32)

        maps = {
            "zeff": self.test_zeff,
            "rhoe": getattr(self, "test_rhoe_absolute", self.test_rhoe),
            "ratio_low_high": ratio,
            "mu_low": mu_low,
            "mu_high": mu_high,
        }
        mask = getattr(self, "test_valid_mask", None)
        if mask is not None:
            maps["mask"] = np.asarray(mask, dtype=np.float32)

        source_name = ""
        try:
            source_name = os.path.basename(self.low_files[index])
        except Exception:
            pass

        saved = []
        try:
            for name, array in maps.items():
                path = os.path.join(
                    folder,
                    "Geo_DECT_slice{:05d}_{}.tif".format(index, name),
                )
                tf.imwrite(path, np.asarray(array, dtype=np.float32))
                saved.append(os.path.basename(path))
        except Exception as error:
            QtWidgets.QMessageBox.critical(
                self,
                "Geo_DECT \u2014 Save test slice failed",
                str(error),
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            "Geo_DECT \u2014 Test slice saved",
            "Slice index {} (LOW file: {})\n\nSaved in:\n{}\n\n{}".format(
                index, source_name or "?", folder, "\n".join(saved)
            ),
        )

    @staticmethod
    def _live_slider_to_value(position, full_minimum, full_maximum):
        fraction = float(position) / 10000.0
        return float(full_minimum) + fraction * (
            float(full_maximum) - float(full_minimum)
        )

    @staticmethod
    def _live_value_to_slider(value, full_minimum, full_maximum):
        denominator = max(float(full_maximum) - float(full_minimum), 1.0e-12)
        fraction = (float(value) - float(full_minimum)) / denominator
        return int(round(np.clip(fraction, 0.0, 1.0) * 10000.0))

    def _show_live_display_range_dock(self):
        """Create a compact Dragonfly-style right dock for live ranges."""
        settings = getattr(self, "geodect_3d_display_settings", {})
        if not settings:
            return

        if self._display_range_dock is not None:
            try:
                self._display_range_dock.show()
                self._display_range_dock.raise_()
                return
            except Exception:
                self._display_range_dock = None

        host = self.window()
        while host.parentWidget() is not None:
            host = host.parentWidget()

        dock = QtWidgets.QDockWidget("Geo_DECT display range", host)
        dock.setObjectName("GeoDECTLiveDisplayRangeDock")
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        body = QtWidgets.QWidget(dock)
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        note = QtWidgets.QLabel(
            "The blue track spans the positive minimum and maximum of the "
            "complete reconstructed stack. Changes are applied automatically "
            "to both 2D and 3D."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        selector_row = QtWidgets.QHBoxLayout()
        selector_row.addWidget(QtWidgets.QLabel("Active image:"))
        property_selector = QtWidgets.QComboBox()
        property_selector.addItem("Z_eff", "zeff")
        property_selector.addItem("rho_e", "rhoe")
        property_selector.addItem("Combined (Z_eff / rho_e)", "combined")
        selector_row.addWidget(property_selector, 1)
        layout.addLayout(selector_row)

        property_stack = QtWidgets.QStackedWidget()
        layout.addWidget(property_stack)

        controls = {
            "property_selector": property_selector,
            "property_stack": property_stack,
        }

        def add_property(key, title_html, full_minimum, full_maximum,
                         selected_minimum, selected_maximum, decimals):
            # Plain-text display name (no rich text) for buttons, dialog
            # titles and status messages, which cannot render <i>/<sub>.
            # Matches the convention already used everywhere else in this
            # plugin for plain text ("Geo_DECT -- Z_eff (effective atomic
            # number)", "Force 3D coloring (Z_eff / rho_e)", etc.) instead
            # of the raw lowercase dict key ("zeff"/"rhoe").
            display_name = "Z_eff" if key == "zeff" else "rho_e"

            container = QtWidgets.QWidget()
            container_layout = QtWidgets.QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(4)

            title_label = QtWidgets.QLabel(title_html)
            title_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
            container_layout.addWidget(title_label)

            group = QtWidgets.QGroupBox()
            grid = QtWidgets.QGridLayout(group)
            container_layout.addWidget(group)

            full_label = QtWidgets.QLabel(
                "Stack: {} – {}".format(
                    self._format_number(full_minimum),
                    self._format_number(full_maximum),
                )
            )
            full_label.setWordWrap(True)
            grid.addWidget(full_label, 0, 0, 1, 2)

            minimum_spin = ScientificDoubleSpinBox()
            maximum_spin = ScientificDoubleSpinBox()
            for spin in (minimum_spin, maximum_spin):
                spin.setDecimals(decimals)
                spin.setRange(float(full_minimum), float(full_maximum))
                spin.setKeyboardTracking(False)
                if key == "rhoe":
                    # The channel/spin values are stored divided by
                    # rhoe_display_scale (~1e23) so the legend doesn't
                    # show 24-digit numbers -- the suffix makes that
                    # explicit instead of silently showing a plain
                    # number that looks like the true electron density.
                    spin.setSuffix(" \u00d710\u00b2\u00b3 e\u207b\u00b7cm\u207b\u00b3")

            minimum_spin.setValue(float(selected_minimum))
            maximum_spin.setValue(float(selected_maximum))

            minimum_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            maximum_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            for slider in (minimum_slider, maximum_slider):
                slider.setRange(0, 10000)
                slider.setStyleSheet(
                    "QSlider::groove:horizontal {height: 7px; "
                    "background: #168de2; border-radius: 3px;} "
                    "QSlider::handle:horizontal {width: 14px; margin: -5px 0; "
                    "background: #e8e8e8; border: 1px solid #666; "
                    "border-radius: 7px;}"
                )

            minimum_slider.setValue(self._live_value_to_slider(
                selected_minimum, full_minimum, full_maximum
            ))
            maximum_slider.setValue(self._live_value_to_slider(
                selected_maximum, full_minimum, full_maximum
            ))

            grid.addWidget(QtWidgets.QLabel("Minimum"), 1, 0)
            grid.addWidget(minimum_spin, 1, 1)
            grid.addWidget(minimum_slider, 2, 0, 1, 2)
            grid.addWidget(QtWidgets.QLabel("Maximum"), 3, 0)
            grid.addWidget(maximum_spin, 3, 1)
            grid.addWidget(maximum_slider, 4, 0, 1, 2)

            # Dragonfly's own native 2D scale bar reads the internal
            # integer display codes (0-4095), not the true physical
            # value, once a property is baked with encode_scalar_slice --
            # so it shows meaningless numbers. This colorbar is generated
            # by the plugin itself directly from the palette and the
            # applied range, so it always reads correctly in real units.
            # Click to enlarge and save as an image (for papers/slides).
            colorbar_label = QtWidgets.QLabel()
            colorbar_label.setCursor(
                QtCore.Qt.CursorShape.PointingHandCursor
            )
            colorbar_label.setToolTip("Click to enlarge and save.")
            grid.addWidget(colorbar_label, 5, 0, 1, 2)

            colorbar_state = {"pixmap": None}

            def refresh_colorbar_preview(range_minimum, range_maximum):
                try:
                    colorbar_rgb = (
                        create_zeff_colorbar(range_minimum, range_maximum)
                        if key == "zeff"
                        else create_rhoe_colorbar(range_minimum, range_maximum)
                    )
                    unit_text = (
                        "\u00d710\u00b2\u00b3 e\u207b\u00b7cm\u207b\u00b3"
                        if key == "rhoe" else ""
                    )
                    labeled_pixmap = self._create_labeled_colorbar_pixmap(
                        colorbar_rgb, range_minimum, range_maximum,
                        unit_suffix=unit_text,
                    )
                    colorbar_state["pixmap"] = labeled_pixmap
                    colorbar_state["range"] = (range_minimum, range_maximum)
                    colorbar_state["unit_text"] = unit_text
                    # No further scaling here -- the pixmap is already
                    # built at its real display size with a legible
                    # font, so shrinking it back down (as before) is
                    # exactly what was making the numbers unreadable.
                    colorbar_label.setFixedSize(labeled_pixmap.size())
                    colorbar_label.setPixmap(labeled_pixmap)
                except Exception:
                    pass

            def show_enlarged_colorbar():
                if colorbar_state.get("pixmap") is None:
                    return
                # Rebuild at high resolution for the enlarged/save view
                # instead of just stretching the small pixmap, which
                # would come out blurry -- this one stays crisp enough
                # for a paper figure or a slide.
                try:
                    range_minimum, range_maximum = colorbar_state["range"]
                    large_rgb = (
                        create_zeff_colorbar(range_minimum, range_maximum)
                        if key == "zeff"
                        else create_rhoe_colorbar(range_minimum, range_maximum)
                    )
                    large_pixmap = self._create_labeled_colorbar_pixmap(
                        large_rgb, range_minimum, range_maximum,
                        unit_suffix=colorbar_state.get("unit_text", ""),
                        tick_count=6,
                        target_width=900,
                        gradient_height=90,
                        tick_font_size=20,
                    )
                except Exception:
                    large_pixmap = colorbar_state["pixmap"]
                self._show_enlarged_pixmap_dialog(
                    "Geo_DECT — {} colorbar".format(display_name),
                    large_pixmap,
                    "Geo_DECT_{}_colorbar.png".format(key),
                    max_size=1000,
                )

            colorbar_label.mousePressEvent = (
                lambda _event: show_enlarged_colorbar()
            )

            refresh_colorbar_preview(
                float(selected_minimum), float(selected_maximum)
            )

            reset_button = QtWidgets.QPushButton("Full stack range")
            grid.addWidget(reset_button, 7, 0, 1, 2)

            apply_range_button = QtWidgets.QPushButton(
                "Apply (recompute {} volume)".format(display_name)
            )
            grid.addWidget(apply_range_button, 8, 0, 1, 2)

            export_stats_button = QtWidgets.QPushButton(
                "Export statistics (PDF)..."
            )
            grid.addWidget(export_stats_button, 9, 0, 1, 2)

            property_stack.addWidget(container)

            state = {
                "full_minimum": float(full_minimum),
                "full_maximum": float(full_maximum),
                "minimum_spin": minimum_spin,
                "maximum_spin": maximum_spin,
                "minimum_slider": minimum_slider,
                "maximum_slider": maximum_slider,
            }
            controls[key] = state

            updating = {"active": False}

            def slider_changed(which):
                if updating["active"]:
                    return
                updating["active"] = True
                try:
                    slider = minimum_slider if which == "minimum" else maximum_slider
                    spin = minimum_spin if which == "minimum" else maximum_spin
                    spin.setValue(self._live_slider_to_value(
                        slider.value(), full_minimum, full_maximum
                    ))
                    if minimum_spin.value() >= maximum_spin.value():
                        if which == "minimum":
                            maximum_spin.setValue(min(
                                full_maximum,
                                minimum_spin.value() + max(
                                    (full_maximum - full_minimum) / 10000.0,
                                    10.0 ** (-decimals),
                                ),
                            ))
                        else:
                            minimum_spin.setValue(max(
                                full_minimum,
                                maximum_spin.value() - max(
                                    (full_maximum - full_minimum) / 10000.0,
                                    10.0 ** (-decimals),
                                ),
                            ))
                finally:
                    updating["active"] = False

            def spin_changed(which):
                if updating["active"]:
                    return
                updating["active"] = True
                try:
                    spin = minimum_spin if which == "minimum" else maximum_spin
                    slider = minimum_slider if which == "minimum" else maximum_slider
                    slider.setValue(self._live_value_to_slider(
                        spin.value(), full_minimum, full_maximum
                    ))
                finally:
                    updating["active"] = False

            minimum_slider.valueChanged.connect(
                lambda _value: slider_changed("minimum")
            )
            maximum_slider.valueChanged.connect(
                lambda _value: slider_changed("maximum")
            )
            minimum_spin.valueChanged.connect(
                lambda _value: spin_changed("minimum")
            )
            maximum_spin.valueChanged.connect(
                lambda _value: spin_changed("maximum")
            )

            def reset_range():
                minimum_spin.setValue(float(full_minimum))
                maximum_spin.setValue(float(full_maximum))

            reset_button.clicked.connect(reset_range)

            def apply_property_range():
                selected_minimum = minimum_spin.value()
                selected_maximum = maximum_spin.value()
                if selected_maximum <= selected_minimum:
                    self.volume_status_label.setText(
                        "Apply for {} skipped: maximum ({}) is not "
                        "greater than minimum ({}).".format(
                            display_name, selected_maximum, selected_minimum
                        )
                    )
                    return

                raw_arrays = getattr(self, "geodect_3d_raw_arrays", {})
                channels = getattr(self, "geodect_3d_channels", {})
                visuals = getattr(self, "geodect_3d_visuals", {})
                channel = channels.get(key)
                visual = visuals.get(key)
                scene_layout = getattr(self, "geodect_3d_scene_layout", None)
                if (
                    key not in raw_arrays
                    or channel is None
                    or visual is None
                    or scene_layout is None
                ):
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Apply — {}".format(display_name),
                        "Not applied: the 3D volumes do not appear to be "
                        "built yet in this session.",
                    )
                    return

                apply_range_button.setEnabled(False)
                try:
                    from . import geodect_3d as geodect3d

                    # Same philosophy as "combined": whether a voxel is
                    # inside/outside the selected range is baked directly
                    # into the CHANNEL'S OWN DATA (as integer display
                    # codes), not left to the 3D Window/Level to decide --
                    # confirmed the only reliable way to also exclude
                    # out-of-range voxels in the 3D view, not just 2D.
                    self.geodect_3d_display_settings[f"{key}_minimum"] = (
                        selected_minimum
                    )
                    self.geodect_3d_display_settings[f"{key}_maximum"] = (
                        selected_maximum
                    )

                    palette = (
                        ZEFF_PALETTE_RGB255 if key == "zeff"
                        else RHOE_PALETTE_RGB255
                    )
                    palette = np.asarray(palette, dtype=np.float64) / 255.0

                    raw_array = raw_arrays[key]
                    channel_array = channel.getNDArray()
                    depth = channel_array.shape[0]

                    for z_index in range(depth):
                        value_slice = raw_array[z_index]
                        valid = value_slice > 0.0
                        codes_slice = geodect3d.encode_scalar_slice(
                            value_slice,
                            valid,
                            selected_minimum,
                            selected_maximum,
                        )
                        channel_array[z_index] = codes_slice

                    channel.setDataDirty()

                    # The selected range is now baked into the channel's
                    # own data as discrete codes -- tag the channel so a
                    # future reconnect (after a project save/reload)
                    # knows it no longer needs the continuous range at
                    # all, just the fixed discrete LUT.
                    geodect3d.persist_display_settings(
                        channel, self.geodect_3d_display_settings, "discrete"
                    )

                    # Same discrete LUT drives both 2D and 3D -- matching
                    # "combined", which never needs a continuous variant.
                    lut = geodect3d.create_scalar_context_lut(
                        "Geo_DECT {} display".format(key),
                        palette,
                    )

                    table_size = (
                        geodect3d.SCALAR_FIRST_COLOR_CODE
                        + geodect3d.SCALAR_BINS
                    )
                    width = float(table_size - 1)
                    center = float(width / 2.0)

                    LayoutPropertiesHelper.setWindowLevel2D(
                        scene_layout, visual, width, center, logging=False
                    )
                    lut_result = geodect3d.apply_lut_to_channel_in_layout(
                        channel=channel,
                        lut=lut,
                        scene_layout=scene_layout,
                        visual_channel=visual,
                        view=getattr(self, "geodect_3d_current_view", None),
                    )
                    if lut_result != "OK":
                        raise RuntimeError(
                            "2D LUT application returned {!r} instead of "
                            "'OK'.".format(lut_result)
                        )
                    LayoutPropertiesHelper.setWindowLevel2D(
                        scene_layout, visual, width, center, logging=False
                    )
                    self.geodect_3d_luts[key] = lut

                    visual.setPropertyDirty()
                    visual.refreshAllParentViews()
                    view = getattr(self, "geodect_3d_current_view", None)
                    if view is not None:
                        try:
                            LayoutHelper.refreshAllViewsInSceneLayoutOfView(
                                view
                            )
                        except Exception:
                            view.refresh()

                    LayoutPropertiesHelper.setWindowLevel3D(
                        scene_layout, visual, width, center, logging=False
                    )
                    geodect3d.apply_confirmed_3d_lut(
                        scene_layout, visual, lut
                    )

                    # Best-effort only, never allowed to affect the
                    # coloring above: hide Dragonfly's own native 2D
                    # scale bar and legend, which (now that the channel
                    # holds integer display codes) can only ever show
                    # meaningless "0-4095" numbers -- the plugin's own
                    # colorbar (updated below) is the correct reference.
                    # Confirmed working: needs an explicit 2D view
                    # refresh right after setting both properties.
                    try:
                        from OrsHelpers.viewLayoutHelper import (
                            DisplayLayoutHelper,
                        )

                        layout_name = scene_layout.getGenealogicalName()
                        LayoutPropertiesHelper.setShowScaleBarFromGenealogicalName(
                            layout_name, False
                        )
                        LayoutPropertiesHelper.setShowLegendFromGenealogicalName(
                            layout_name, False
                        )
                        for a_view in DisplayLayoutHelper.getAll2DViewsInLayout(
                            scene_layout
                        ):
                            a_view.refresh()
                    except Exception:
                        pass

                    refresh_colorbar_preview(selected_minimum, selected_maximum)

                    self.volume_status_label.setText(
                        "{} range applied ({} - {}).".format(
                            display_name,
                            self._format_number(selected_minimum),
                            self._format_number(selected_maximum),
                        )
                    )
                except Exception as exc:
                    self.volume_status_label.setText(
                        "Apply for {} failed: {}: {}".format(
                            display_name, type(exc).__name__, exc
                        )
                    )
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Apply — {} failed".format(display_name),
                        "{}: {}".format(type(exc).__name__, exc),
                    )
                finally:
                    apply_range_button.setEnabled(True)

            apply_range_button.clicked.connect(apply_property_range)

            def export_statistics_pdf():
                raw_arrays = getattr(self, "geodect_3d_raw_arrays", {})
                if key not in raw_arrays:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Export statistics — {}".format(display_name),
                        "Not available: the 3D volumes are not built yet.",
                    )
                    return

                settings = self.geodect_3d_display_settings
                # Always use the range most recently confirmed with
                # "Apply" -- not whatever the slider currently shows if
                # the user moved it again without clicking Apply.
                applied_minimum = settings.get(
                    f"{key}_minimum", minimum_spin.value()
                )
                applied_maximum = settings.get(
                    f"{key}_maximum", maximum_spin.value()
                )

                rhoe_display_scale = float(
                    settings.get("rhoe_display_scale", 1.0)
                )
                # raw_arrays["rhoe"] and the settings min/max are stored
                # in DISPLAY units (divided by rhoe_display_scale, e.g.
                # 1e23, so the 3D legend doesn't show 24-digit numbers).
                # Statistics must always report the true physical value
                # with the correct unit -- multiply back here, the one
                # place all the numbers funnel through before display.
                if key == "rhoe":
                    scale, unit_suffix = self._rhoe_report_units(
                        rhoe_display_scale
                    )
                    property_label = r"$\rho_e$"
                else:
                    scale = 1.0
                    property_label = r"$Z_{eff}$"
                    unit_suffix = ""

                # Options dialog: number of populations and exclusion of
                # interface (partial-volume) voxels, which otherwise fill
                # the valley between two real phases in the histogram.
                options_dialog = QtWidgets.QDialog(self)
                options_dialog.setWindowTitle(
                    "Export statistics \u2014 {}".format(display_name)
                )
                options_layout = QtWidgets.QFormLayout(options_dialog)

                populations_combo = QtWidgets.QComboBox()
                populations_combo.addItem("Automatic (BIC, 1\u20133)", None)
                populations_combo.addItem("1", 1)
                populations_combo.addItem("2", 2)
                populations_combo.addItem("3", 3)
                options_layout.addRow(
                    "Number of populations", populations_combo
                )

                exclude_interfaces_check = QtWidgets.QCheckBox(
                    "Exclude interface (partial-volume) voxels"
                )
                exclude_interfaces_check.setChecked(True)
                options_layout.addRow(exclude_interfaces_check)

                gradient_spin = QtWidgets.QSpinBox()
                gradient_spin.setRange(0, 80)
                gradient_spin.setValue(30)
                gradient_spin.setSuffix(" %")
                gradient_spin.setToolTip(
                    "Per slice, drop this percentage of voxels with the "
                    "strongest local gradient (voxels on phase boundaries)."
                )
                options_layout.addRow(
                    "Strongest-gradient voxels removed", gradient_spin
                )

                def _toggle_exclusion(checked):
                    gradient_spin.setEnabled(checked)

                exclude_interfaces_check.toggled.connect(_toggle_exclusion)

                options_buttons = QtWidgets.QDialogButtonBox(
                    QtWidgets.QDialogButtonBox.StandardButton.Ok
                    | QtWidgets.QDialogButtonBox.StandardButton.Cancel
                )
                options_buttons.accepted.connect(options_dialog.accept)
                options_buttons.rejected.connect(options_dialog.reject)
                options_layout.addRow(options_buttons)

                if options_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                    return

                n_components = populations_combo.currentData()
                exclude_interfaces = exclude_interfaces_check.isChecked()
                gradient_percent = (
                    gradient_spin.value() if exclude_interfaces else 0
                )
                # Edge erosion is NOT repeated here: the retained-voxel
                # mask is already eroded at Build time ("Mask erosion").
                edge_px = 0

                self.volume_status_label.setText(
                    "Collecting {} voxels for statistics...".format(
                        display_name
                    )
                )
                QtWidgets.QApplication.processEvents()

                from . import geodect_statistics as geodect_stats

                values, total_valid, excluded_count = (
                    geodect_stats.select_interior_values(
                        raw_arrays[key],
                        scale=scale,
                        exclude_gradient_percent=gradient_percent,
                        edge_erosion_px=edge_px,
                    )
                )
                selected_minimum = float(applied_minimum) * scale
                selected_maximum = float(applied_maximum) * scale

                if exclude_interfaces and total_valid:
                    exclusion_note = (
                        "Interface voxels excluded: {:,} of {:,} ({:.1f}%) "
                        "\u2014 strongest {}% local gradients per slice.".format(
                            excluded_count, total_valid,
                            100.0 * excluded_count / total_valid,
                            gradient_percent,
                        )
                    )
                else:
                    exclusion_note = "All valid voxels included."

                if values.size == 0:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Export statistics — {}".format(display_name),
                        "No valid voxels found for {}.".format(display_name),
                    )
                    return

                default_name = "Geo_DECT_{}_statistics.pdf".format(key)
                file_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Export {} statistics as PDF".format(display_name),
                    default_name,
                    "PDF files (*.pdf)",
                )
                if not file_path:
                    return
                if not file_path.lower().endswith(".pdf"):
                    file_path += ".pdf"

                export_stats_button.setEnabled(False)
                try:
                    self.volume_status_label.setText(
                        "Computing {} statistics... this can take a "
                        "moment for large volumes.".format(display_name)
                    )
                    QtWidgets.QApplication.processEvents()

                    from . import geodect_statistics as geodect_stats

                    figure, _stats = geodect_stats.create_scalar_statistics_figure(
                        values,
                        selected_minimum,
                        selected_maximum,
                        property_label,
                        unit_suffix=unit_suffix,
                        n_components=n_components,
                        exclusion_note=exclusion_note,
                    )
                    figure.savefig(file_path)
                    import matplotlib.pyplot as plt
                    plt.close(figure)

                    self.volume_status_label.setText(
                        "{} statistics exported to {}".format(
                            display_name, file_path
                        )
                    )
                    QtGui.QDesktopServices.openUrl(
                        QtCore.QUrl.fromLocalFile(file_path)
                    )
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Export statistics — {} failed".format(display_name),
                        "{}: {}".format(type(exc).__name__, exc),
                    )
                finally:
                    export_stats_button.setEnabled(True)

            export_stats_button.clicked.connect(export_statistics_pdf)

        add_property(
            "zeff",
            "<div style='font-size:12pt; font-weight:700;'>"
            "<i>Z<sub>eff</sub></i></div>",
            settings["zeff_full_minimum"],
            settings["zeff_full_maximum"],
            settings["zeff_minimum"],
            settings["zeff_maximum"],
            4,
        )
        add_property(
            "rhoe",
            "<div style='font-size:12pt; font-weight:700;'>"
            "<i>\u03c1<sub>e</sub></i></div>",
            settings["rhoe_full_minimum"],
            settings["rhoe_full_maximum"],
            settings["rhoe_minimum"],
            settings["rhoe_maximum"],
            6,
        )

        # --- Combined page: interactive 2D colorimetric range selector ---
        combined_container = QtWidgets.QWidget()
        combined_container_layout = QtWidgets.QVBoxLayout(combined_container)
        combined_container_layout.setContentsMargins(0, 0, 0, 0)
        combined_container_layout.setSpacing(4)

        combined_title_label = QtWidgets.QLabel(
            "<div style='font-size:12pt; font-weight:700;'>"
            "Combined (<i>Z<sub>eff</sub></i> / <i>\u03c1<sub>e</sub></i>)"
            "</div>"
        )
        combined_title_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        combined_container_layout.addWidget(combined_title_label)

        combined_group = QtWidgets.QGroupBox()
        combined_layout = QtWidgets.QVBoxLayout(combined_group)
        combined_container_layout.addWidget(combined_group)

        rhoe_display_scale = float(settings.get("rhoe_display_scale", 1.0))

        scatter = GeoDECTScatterRangeSelector(
            settings["zeff_full_minimum"],
            settings["zeff_full_maximum"],
            settings["rhoe_full_minimum"],
            settings["rhoe_full_maximum"],
            settings["zeff_full_minimum"],
            settings["zeff_full_maximum"],
            settings["rhoe_full_minimum"],
            settings["rhoe_full_maximum"],
            rhoe_display_scale=rhoe_display_scale,
        )

        scatter_row = QtWidgets.QHBoxLayout()
        scatter_row.addWidget(scatter, 1)

        redistribution_preview_state = {
            "zeff_min": 0.0, "zeff_max": 1.0,
            "rhoe_min": 0.0, "rhoe_max": 1.0,
        }

        preview_column = QtWidgets.QVBoxLayout()
        preview_label = QtWidgets.QLabel(
            "Preview: colours after\nredistribution (Apply)"
        )
        preview_label.setWordWrap(True)
        preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        preview_column.addWidget(preview_label)
        redistribution_preview = GeoDECTRedistributionPreview()
        redistribution_preview.setCursor(
            QtCore.Qt.CursorShape.PointingHandCursor
        )
        redistribution_preview.setToolTip("Click to enlarge and save.")
        preview_column.addWidget(redistribution_preview)

        def show_enlarged_redistribution_preview():
            state = redistribution_preview_state
            if state["zeff_max"] <= state["zeff_min"] or (
                state["rhoe_max"] <= state["rhoe_min"]
            ):
                return
            try:
                large_rgb = create_geodect_legend(
                    state["zeff_min"], state["zeff_max"],
                    state["rhoe_min"], state["rhoe_max"],
                    width=640, height=640,
                )
                large_pixmap = self._numpy_rgb_to_pixmap(large_rgb)
            except Exception:
                return
            self._show_enlarged_pixmap_dialog(
                "Geo_DECT — combined redistribution preview",
                large_pixmap,
                "Geo_DECT_combined_redistribution_preview.png",
                max_size=700,
            )

        redistribution_preview.mousePressEvent = (
            lambda _event: show_enlarged_redistribution_preview()
        )

        combined_readout = QtWidgets.QLabel()
        combined_readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        combined_readout.setWordWrap(True)
        preview_column.addWidget(combined_readout)

        preview_column.addStretch(1)
        scatter_row.addLayout(preview_column)

        combined_layout.addLayout(scatter_row, 1)

        # --- Manual min/max entry for the combined redistribution ---
        # Dragging the rectangle on the scatterplot is fine for a rough
        # selection, but picking an exact boundary by hand is hard with
        # a mouse -- these four spin+slider pairs (same style as the
        # single-property Z_eff/rho_e panels above) let the user type
        # the desired Z_eff and rho_e min/max directly. Kept in sync
        # both ways with the scatterplot's own rectangle: typing here
        # moves the rectangle, dragging the rectangle updates these
        # values (see refresh_combined_readout below).
        manual_range_group = QtWidgets.QGroupBox("Manual range entry")
        manual_range_grid = QtWidgets.QGridLayout(manual_range_group)
        combined_layout.addWidget(manual_range_group)

        manual_range_state = {"updating": False}

        def _make_manual_spin(decimals, full_minimum, full_maximum, suffix=""):
            spin = ScientificDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(float(full_minimum), float(full_maximum))
            spin.setKeyboardTracking(False)
            if suffix:
                spin.setSuffix(suffix)
            return spin

        # Two decimals is plenty for Z_eff here -- this is purely about
        # picking a display/redistribution boundary, not the underlying
        # calculation, which keeps its full precision regardless.
        zeff_min_spin = _make_manual_spin(
            2, settings["zeff_full_minimum"], settings["zeff_full_maximum"]
        )
        zeff_max_spin = _make_manual_spin(
            2, settings["zeff_full_minimum"], settings["zeff_full_maximum"]
        )
        rhoe_min_spin = _make_manual_spin(
            4, settings["rhoe_full_minimum"], settings["rhoe_full_maximum"],
            suffix=" \u00d710\u00b2\u00b3 e\u207b\u00b7cm\u207b\u00b3",
        )
        rhoe_max_spin = _make_manual_spin(
            4, settings["rhoe_full_minimum"], settings["rhoe_full_maximum"],
            suffix=" \u00d710\u00b2\u00b3 e\u207b\u00b7cm\u207b\u00b3",
        )

        def _make_manual_slider():
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, 10000)
            slider.setStyleSheet(
                "QSlider::groove:horizontal {height: 7px; "
                "background: #168de2; border-radius: 3px;} "
                "QSlider::handle:horizontal {width: 14px; margin: -5px 0; "
                "background: #e8e8e8; border: 1px solid #666; "
                "border-radius: 7px;}"
            )
            return slider

        zeff_min_slider = _make_manual_slider()
        zeff_max_slider = _make_manual_slider()
        rhoe_min_slider = _make_manual_slider()
        rhoe_max_slider = _make_manual_slider()

        def _make_manual_label(html_text):
            label = QtWidgets.QLabel(html_text)
            label.setTextFormat(QtCore.Qt.TextFormat.RichText)
            return label

        manual_range_grid.addWidget(
            _make_manual_label("<i>Z<sub>eff</sub></i> minimum"), 0, 0
        )
        manual_range_grid.addWidget(zeff_min_spin, 0, 1)
        manual_range_grid.addWidget(zeff_min_slider, 1, 0, 1, 2)
        manual_range_grid.addWidget(
            _make_manual_label("<i>Z<sub>eff</sub></i> maximum"), 2, 0
        )
        manual_range_grid.addWidget(zeff_max_spin, 2, 1)
        manual_range_grid.addWidget(zeff_max_slider, 3, 0, 1, 2)
        manual_range_grid.addWidget(
            _make_manual_label("<i>\u03c1<sub>e</sub></i> minimum"), 0, 2
        )
        manual_range_grid.addWidget(rhoe_min_spin, 0, 3)
        manual_range_grid.addWidget(rhoe_min_slider, 1, 2, 1, 2)
        manual_range_grid.addWidget(
            _make_manual_label("<i>\u03c1<sub>e</sub></i> maximum"), 2, 2
        )
        manual_range_grid.addWidget(rhoe_max_spin, 2, 3)
        manual_range_grid.addWidget(rhoe_max_slider, 3, 2, 1, 2)

        def _manual_control_changed(_value=None):
            if manual_range_state["updating"]:
                return
            manual_range_state["updating"] = True
            try:
                zeff_min_slider.setValue(self._live_value_to_slider(
                    zeff_min_spin.value(), settings["zeff_full_minimum"],
                    settings["zeff_full_maximum"],
                ))
                zeff_max_slider.setValue(self._live_value_to_slider(
                    zeff_max_spin.value(), settings["zeff_full_minimum"],
                    settings["zeff_full_maximum"],
                ))
                rhoe_min_slider.setValue(self._live_value_to_slider(
                    rhoe_min_spin.value(), settings["rhoe_full_minimum"],
                    settings["rhoe_full_maximum"],
                ))
                rhoe_max_slider.setValue(self._live_value_to_slider(
                    rhoe_max_spin.value(), settings["rhoe_full_minimum"],
                    settings["rhoe_full_maximum"],
                ))
            finally:
                manual_range_state["updating"] = False

            zeff_min = zeff_min_spin.value()
            zeff_max = zeff_max_spin.value()
            rhoe_min = rhoe_min_spin.value()
            rhoe_max = rhoe_max_spin.value()
            if zeff_min >= zeff_max:
                zeff_max = min(
                    settings["zeff_full_maximum"], zeff_min + 0.01
                )
            if rhoe_min >= rhoe_max:
                rhoe_max = min(
                    settings["rhoe_full_maximum"], rhoe_min + 1e-4
                )
            scatter.set_selection(
                zeff_min, zeff_max, rhoe_min, rhoe_max, emit_signal=True
            )

        def _manual_slider_changed(spin, slider, full_minimum, full_maximum):
            if manual_range_state["updating"]:
                return
            manual_range_state["updating"] = True
            try:
                spin.setValue(self._live_slider_to_value(
                    slider.value(), full_minimum, full_maximum
                ))
            finally:
                manual_range_state["updating"] = False
            _manual_control_changed()

        zeff_min_spin.valueChanged.connect(_manual_control_changed)
        zeff_max_spin.valueChanged.connect(_manual_control_changed)
        rhoe_min_spin.valueChanged.connect(_manual_control_changed)
        rhoe_max_spin.valueChanged.connect(_manual_control_changed)
        zeff_min_slider.valueChanged.connect(
            lambda _v: _manual_slider_changed(
                zeff_min_spin, zeff_min_slider,
                settings["zeff_full_minimum"], settings["zeff_full_maximum"],
            )
        )
        zeff_max_slider.valueChanged.connect(
            lambda _v: _manual_slider_changed(
                zeff_max_spin, zeff_max_slider,
                settings["zeff_full_minimum"], settings["zeff_full_maximum"],
            )
        )
        rhoe_min_slider.valueChanged.connect(
            lambda _v: _manual_slider_changed(
                rhoe_min_spin, rhoe_min_slider,
                settings["rhoe_full_minimum"], settings["rhoe_full_maximum"],
            )
        )
        rhoe_max_slider.valueChanged.connect(
            lambda _v: _manual_slider_changed(
                rhoe_max_spin, rhoe_max_slider,
                settings["rhoe_full_minimum"], settings["rhoe_full_maximum"],
            )
        )

        enlarge_scatter_button = QtWidgets.QPushButton(
            "Enlarge / Save..."
        )
        combined_layout.addWidget(enlarge_scatter_button)

        def show_enlarged_scatter():
            zeff_min, zeff_max, rhoe_min, rhoe_max = scatter.get_selection()
            try:
                # A fresh, larger instance -- not the live interactive
                # widget itself -- so this can be grabbed at high
                # resolution without disturbing the on-screen selector.
                # Carries over the current selection AND every probed
                # point, so the saved image matches what is on screen.
                large_scatter = GeoDECTScatterRangeSelector(
                    settings["zeff_full_minimum"],
                    settings["zeff_full_maximum"],
                    settings["rhoe_full_minimum"],
                    settings["rhoe_full_maximum"],
                    zeff_min, zeff_max, rhoe_min, rhoe_max,
                    rhoe_display_scale=rhoe_display_scale,
                )
                for point_zeff, point_rhoe in getattr(
                    scatter, "_points", []
                ):
                    large_scatter.add_point(point_zeff, point_rhoe)
                large_scatter.resize(900, 900)
                scatter_pixmap = large_scatter.grab()
                # Same font size the left scatterplot's own axes just
                # used, read straight from it -- so both squares' axis
                # text end up genuinely uniform instead of two
                # separately-guessed sizes that never quite matched.
                tick_font_size = max(
                    8, min(18, round(8 * large_scatter._size_scale() ** 0.7))
                )
                title_font_size = min(tick_font_size + 5, 22)

                # Same "redistributed colours" square shown next to the
                # scatterplot in the panel, rebuilt at high resolution --
                # composed together into ONE image, exactly like the two
                # squares appear side by side on screen. Axis labels
                # (Z_eff along the bottom, rho_e along the left) use the
                # same compact 1-decimal formatting and font size as the
                # main scatterplot -- no title, no separate readout text
                # underneath, both dropped as no longer needed once the
                # axes themselves carry the numbers.
                preview_size = 550
                preview_margin_left = 130
                preview_margin_bottom = 70
                preview_pixmap = None
                if zeff_max > zeff_min and rhoe_max > rhoe_min:
                    preview_rgb = create_geodect_legend(
                        zeff_min, zeff_max, rhoe_min, rhoe_max,
                        width=preview_size, height=preview_size,
                    )
                    preview_pixmap = self._numpy_rgb_to_pixmap(preview_rgb)

                gap = 50
                right_column_width = (
                    preview_margin_left + preview_size + 20
                )
                total_width = (
                    scatter_pixmap.width() + gap + right_column_width
                )
                total_height = max(
                    scatter_pixmap.height(),
                    preview_size + preview_margin_bottom,
                )

                composed = QtGui.QPixmap(total_width, total_height)
                composed.fill(QtCore.Qt.GlobalColor.black)
                painter = QtGui.QPainter(composed)
                painter.setRenderHint(
                    QtGui.QPainter.RenderHint.Antialiasing, True
                )
                painter.drawPixmap(0, 0, scatter_pixmap)

                right_x = scatter_pixmap.width() + gap
                preview_top = (
                    total_height - preview_margin_bottom - preview_size
                ) // 2
                preview_left = right_x + preview_margin_left

                if preview_pixmap is not None:
                    painter.drawPixmap(
                        preview_left, preview_top, preview_pixmap
                    )
                    border_pen = QtGui.QPen(
                        QtGui.QColor(255, 255, 255)
                    )
                    border_pen.setWidth(2)
                    border_pen.setStyle(QtCore.Qt.PenStyle.DashLine)
                    painter.setPen(border_pen)
                    painter.drawRect(
                        preview_left, preview_top,
                        preview_size, preview_size,
                    )

                    axis_font = QtGui.QFont()
                    axis_font.setPointSize(tick_font_size)
                    painter.setFont(axis_font)
                    painter.setPen(
                        QtGui.QPen(QtCore.Qt.GlobalColor.white)
                    )

                    zeff_mid = (zeff_min + zeff_max) / 2.0
                    rhoe_mid = (rhoe_min + rhoe_max) / 2.0
                    label_w = tick_font_size * 8

                    # Z_eff along the bottom -- min/median/max.
                    preview_bottom = preview_top + preview_size
                    for fraction, value, align in (
                        (0.0, zeff_min, QtCore.Qt.AlignmentFlag.AlignLeft),
                        (0.5, zeff_mid, QtCore.Qt.AlignmentFlag.AlignHCenter),
                        (1.0, zeff_max, QtCore.Qt.AlignmentFlag.AlignRight),
                    ):
                        x = preview_left + fraction * preview_size
                        painter.drawText(
                            QtCore.QRectF(
                                x - label_w / 2, preview_bottom + 4,
                                label_w, tick_font_size + 10,
                            ),
                            align,
                            MainFormGeo_dect._format_tick_value(value),
                        )
                    MainFormGeo_dect._draw_html_centered(
                        painter,
                        "<div style='color:#ffffff; font-size:{0}pt; "
                        "font-weight:700; text-align:center;'>"
                        "<i>Z<sub>eff</sub></i></div>".format(
                            title_font_size
                        ),
                        QtCore.QRectF(
                            preview_left,
                            preview_bottom + tick_font_size + 16,
                            preview_size, title_font_size + 10,
                        ),
                    )

                    # rho_e along the left -- min/median/max, values
                    # grow upward.
                    for fraction, value in (
                        (0.0, rhoe_min), (0.5, rhoe_mid), (1.0, rhoe_max),
                    ):
                        y = preview_bottom - fraction * preview_size
                        painter.drawText(
                            QtCore.QRectF(
                                right_x, y - (tick_font_size + 10) / 2,
                                preview_margin_left - 8,
                                tick_font_size + 10,
                            ),
                            QtCore.Qt.AlignmentFlag.AlignRight
                            | QtCore.Qt.AlignmentFlag.AlignVCenter,
                            MainFormGeo_dect._format_tick_value(
                                value * rhoe_display_scale, scientific=True
                            ),
                        )
                    painter.save()
                    painter.translate(
                        right_x + title_font_size + 4,
                        preview_top + preview_size / 2,
                    )
                    painter.rotate(-90)
                    MainFormGeo_dect._draw_html_centered(
                        painter,
                        "<div style='color:#ffffff; font-size:{0}pt; "
                        "font-weight:700; text-align:center;'>"
                        "<i>\u03c1<sub>e</sub></i></div>".format(
                            title_font_size
                        ),
                        QtCore.QRectF(
                            -preview_size / 2, -(title_font_size + 10) / 2,
                            preview_size, title_font_size + 10,
                        ),
                    )
                    painter.restore()

                painter.end()
            except Exception:
                return
            self._show_enlarged_pixmap_dialog(
                "Geo_DECT — combined colorimetric scatterplot",
                composed,
                "Geo_DECT_combined_scatterplot.png",
                max_size=1100,
            )

        enlarge_scatter_button.clicked.connect(show_enlarged_scatter)

        def refresh_combined_readout(zeff_min, zeff_max, rhoe_min, rhoe_max):
            combined_readout.setText(
                "<i>Z<sub>eff</sub></i>: {} \u2013 {}<br>"
                "<i>\u03c1<sub>e</sub></i>: {} \u2013 {} "
                "(\u00d710\u00b2\u00b3)".format(
                    self._format_zeff_readout(zeff_min),
                    self._format_zeff_readout(zeff_max),
                    MainFormGeo_dect._format_tick_value(rhoe_min),
                    MainFormGeo_dect._format_tick_value(rhoe_max),
                )
            )
            redistribution_preview.set_range(
                zeff_min, zeff_max, rhoe_min, rhoe_max
            )
            redistribution_preview_state["zeff_min"] = zeff_min
            redistribution_preview_state["zeff_max"] = zeff_max
            redistribution_preview_state["rhoe_min"] = rhoe_min
            redistribution_preview_state["rhoe_max"] = rhoe_max
            if not manual_range_state["updating"]:
                manual_range_state["updating"] = True
                try:
                    zeff_min_spin.setValue(zeff_min)
                    zeff_max_spin.setValue(zeff_max)
                    rhoe_min_spin.setValue(rhoe_min)
                    rhoe_max_spin.setValue(rhoe_max)
                    zeff_min_slider.setValue(self._live_value_to_slider(
                        zeff_min, settings["zeff_full_minimum"],
                        settings["zeff_full_maximum"],
                    ))
                    zeff_max_slider.setValue(self._live_value_to_slider(
                        zeff_max, settings["zeff_full_minimum"],
                        settings["zeff_full_maximum"],
                    ))
                    rhoe_min_slider.setValue(self._live_value_to_slider(
                        rhoe_min, settings["rhoe_full_minimum"],
                        settings["rhoe_full_maximum"],
                    ))
                    rhoe_max_slider.setValue(self._live_value_to_slider(
                        rhoe_max, settings["rhoe_full_minimum"],
                        settings["rhoe_full_maximum"],
                    ))
                finally:
                    manual_range_state["updating"] = False

        refresh_combined_readout(
            settings["zeff_full_minimum"],
            settings["zeff_full_maximum"],
            settings["rhoe_full_minimum"],
            settings["rhoe_full_maximum"],
        )
        scatter.selectionChanged.connect(refresh_combined_readout)

        combined_apply_button = QtWidgets.QPushButton(
            "Apply (recompute combined volume)"
        )
        combined_layout.addWidget(combined_apply_button)

        def apply_combined_range():
            zeff_min, zeff_max, rhoe_min, rhoe_max = scatter.get_selection()
            if zeff_max <= zeff_min or rhoe_max <= rhoe_min:
                return

            channels = getattr(self, "geodect_3d_channels", {})
            visuals = getattr(self, "geodect_3d_visuals", {})
            zeff_channel = channels.get("zeff")
            rhoe_channel = channels.get("rhoe")
            combined_channel = channels.get("combined")
            combined_visual = visuals.get("combined")
            if None in (
                zeff_channel, rhoe_channel, combined_channel, combined_visual
            ):
                self.volume_status_label.setText(
                    "Combined range not applied: 3D volumes are not "
                    "built yet."
                )
                return

            combined_apply_button.setEnabled(False)
            try:
                from . import geodect_3d as geodect3d

                self.volume_status_label.setText(
                    "Recomputing combined volume..."
                )
                QtWidgets.QApplication.processEvents()

                zeff_array = self.geodect_3d_raw_arrays["zeff"]
                rhoe_array = self.geodect_3d_raw_arrays["rhoe"]
                combined_array = combined_channel.getNDArray()

                depth = combined_array.shape[0]
                for z_index in range(depth):
                    zeff_slice = zeff_array[z_index]
                    rhoe_slice = rhoe_array[z_index]
                    physical = (zeff_slice > 0.0) & (rhoe_slice > 0.0)
                    combined_array[z_index] = geodect3d.encode_geodect_slice(
                        zeff_slice,
                        rhoe_slice,
                        physical,
                        zeff_min,
                        zeff_max,
                        rhoe_min,
                        rhoe_max,
                    )
                    if z_index % 50 == 0:
                        self.volume_status_label.setText(
                            "Recomputing combined volume... "
                            "slice {}/{}".format(z_index + 1, depth)
                        )
                        QtWidgets.QApplication.processEvents()

                combined_channel.setDataDirty()

                self.geodect_3d_display_settings["combined_zeff_minimum"] = zeff_min
                self.geodect_3d_display_settings["combined_zeff_maximum"] = zeff_max
                self.geodect_3d_display_settings["combined_rhoe_minimum"] = rhoe_min
                self.geodect_3d_display_settings["combined_rhoe_maximum"] = rhoe_max

                combined_visual.setPropertyDirty()
                combined_visual.refreshAllParentViews()
                view = getattr(self, "geodect_3d_current_view", None)
                if view is not None:
                    try:
                        LayoutHelper.refreshAllViewsInSceneLayoutOfView(view)
                    except Exception:
                        view.refresh()

                try:
                    geodect3d.apply_confirmed_3d_lut(
                        self.geodect_3d_scene_layout,
                        combined_visual,
                        self.geodect_3d_luts["combined"],
                    )
                except Exception:
                    pass

                # Best-effort only, never allowed to affect the coloring
                # above: keep the native 2D scale bar/legend hidden for
                # the whole layout (reasserted here too in case anything
                # re-enabled it) -- the crossplot export is the correct
                # reference for "combined", the plugin's own colorbars
                # are the reference for Z_eff/rho_e.
                try:
                    from OrsHelpers.viewLayoutHelper import (
                        DisplayLayoutHelper,
                    )

                    scene_layout = self.geodect_3d_scene_layout
                    layout_name = scene_layout.getGenealogicalName()
                    LayoutPropertiesHelper.setShowScaleBarFromGenealogicalName(
                        layout_name, False
                    )
                    LayoutPropertiesHelper.setShowLegendFromGenealogicalName(
                        layout_name, False
                    )
                    for a_view in DisplayLayoutHelper.getAll2DViewsInLayout(
                        scene_layout
                    ):
                        a_view.refresh()
                except Exception:
                    pass

                self.volume_status_label.setText(
                    "Combined volume recomputed for the selected range."
                )
            except Exception as exc:
                self.volume_status_label.setText(
                    "Combined range recompute failed: {}: {}".format(
                        type(exc).__name__, exc
                    )
                )
            finally:
                combined_apply_button.setEnabled(True)

        combined_apply_button.clicked.connect(apply_combined_range)

        export_crossplot_button = QtWidgets.QPushButton(
            "Export crossplot (PDF)..."
        )
        combined_layout.addWidget(export_crossplot_button)

        def export_crossplot_pdf():
            raw_arrays = getattr(self, "geodect_3d_raw_arrays", {})
            if "zeff" not in raw_arrays or "rhoe" not in raw_arrays:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Export crossplot",
                    "Not available: the 3D volumes are not built yet.",
                )
                return

            settings = self.geodect_3d_display_settings
            # The combined volume has its own selected range (from the
            # scatter selector above), independent of the standalone
            # Z_eff/rho_e "Apply" panels -- read the range it was last
            # actually recomputed with.
            zeff_minimum = float(
                settings.get(
                    "combined_zeff_minimum", settings.get("zeff_minimum", 0.0)
                )
            )
            zeff_maximum = float(
                settings.get(
                    "combined_zeff_maximum", settings.get("zeff_maximum", 1.0)
                )
            )
            rhoe_display_scale = float(
                settings.get("rhoe_display_scale", 1.0)
            )
            rhoe_multiplier, rhoe_unit = self._rhoe_report_units(
                rhoe_display_scale
            )
            # rho_e is kept in its stored (display) units, e.g. 10^23
            # e-/cm3, and that multiplier is written in every label.
            rhoe_minimum = float(
                settings.get(
                    "combined_rhoe_minimum", settings.get("rhoe_minimum", 0.0)
                )
            ) * rhoe_multiplier
            rhoe_maximum = float(
                settings.get(
                    "combined_rhoe_maximum", settings.get("rhoe_maximum", 1.0)
                )
            ) * rhoe_multiplier

            options_dialog = QtWidgets.QDialog(self)
            options_dialog.setWindowTitle("Export crossplot")
            options_layout = QtWidgets.QFormLayout(options_dialog)
            populations_combo = QtWidgets.QComboBox()
            populations_combo.addItem("Automatic (BIC, 1\u20133)", None)
            for count in (1, 2, 3):
                populations_combo.addItem(str(count), count)
            options_layout.addRow("Number of populations", populations_combo)
            exclude_interfaces_check = QtWidgets.QCheckBox(
                "Exclude interface (partial-volume) voxels"
            )
            exclude_interfaces_check.setChecked(True)
            options_layout.addRow(exclude_interfaces_check)
            gradient_spin = QtWidgets.QSpinBox()
            gradient_spin.setRange(0, 80)
            gradient_spin.setValue(30)
            gradient_spin.setSuffix(" %")
            gradient_spin.setToolTip(
                "Per slice, drop this percentage of voxels with the "
                "strongest local gradient in Z_eff OR rho_e (voxels on "
                "phase boundaries)."
            )
            options_layout.addRow(
                "Strongest-gradient voxels removed", gradient_spin
            )
            exclude_interfaces_check.toggled.connect(gradient_spin.setEnabled)
            options_buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            options_buttons.accepted.connect(options_dialog.accept)
            options_buttons.rejected.connect(options_dialog.reject)
            options_layout.addRow(options_buttons)
            if options_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            cluster_count = populations_combo.currentData()
            gradient_percent = (
                gradient_spin.value()
                if exclude_interfaces_check.isChecked() else 0
            )

            self.volume_status_label.setText(
                "Collecting Z_eff / rho_e voxels for the crossplot..."
            )
            QtWidgets.QApplication.processEvents()

            from . import geodect_statistics as geodect_stats

            zeff_values, rhoe_values, total_valid, excluded_count = (
                geodect_stats.select_interior_pairs(
                    raw_arrays["zeff"],
                    raw_arrays["rhoe"],
                    exclude_gradient_percent=gradient_percent,
                )
            )
            if rhoe_multiplier != 1.0:
                rhoe_values = rhoe_values.astype(np.float64) * rhoe_multiplier

            if gradient_percent and total_valid:
                exclusion_note = (
                    "Interface voxels excluded: {:,} of {:,} ({:.1f}%) "
                    "\u2014 strongest {}% local gradients per slice."
                    .format(
                        excluded_count, total_valid,
                        100.0 * excluded_count / total_valid,
                        gradient_percent,
                    )
                )
            else:
                exclusion_note = "All valid voxels included."

            if zeff_values.size == 0:
                QtWidgets.QMessageBox.warning(
                    self, "Export crossplot", "No valid voxels found."
                )
                return

            default_name = "Geo_DECT_combined_crossplot.pdf"
            file_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export combined crossplot as PDF",
                default_name,
                "PDF files (*.pdf)",
            )
            if not file_path:
                return
            if not file_path.lower().endswith(".pdf"):
                file_path += ".pdf"

            export_crossplot_button.setEnabled(False)
            try:
                self.volume_status_label.setText(
                    "Computing combined crossplot... this can take a "
                    "moment for large volumes."
                )
                QtWidgets.QApplication.processEvents()

                from . import geodect_statistics as geodect_stats

                figure, _clusters = geodect_stats.create_crossplot_figure(
                    zeff_values,
                    rhoe_values,
                    zeff_minimum,
                    zeff_maximum,
                    rhoe_minimum,
                    rhoe_maximum,
                    n_clusters=cluster_count,
                    rhoe_unit_suffix=rhoe_unit,
                    exclusion_note=exclusion_note,
                )
                figure.savefig(file_path)
                import matplotlib.pyplot as plt
                plt.close(figure)

                self.volume_status_label.setText(
                    "Combined crossplot exported to {}".format(file_path)
                )
                QtGui.QDesktopServices.openUrl(
                    QtCore.QUrl.fromLocalFile(file_path)
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Export crossplot failed",
                    "{}: {}".format(type(exc).__name__, exc),
                )
            finally:
                export_crossplot_button.setEnabled(True)

        export_crossplot_button.clicked.connect(export_crossplot_pdf)

        # --- Probed points: auto-detected from Dragonfly's native Probe
        # tool, shown as dots on the scatter widget, listed in a table,
        # exportable to CSV. ---
        points_label = QtWidgets.QLabel(
            "Probed points (auto-detected from Dragonfly's Probe tool):"
        )
        points_label.setWordWrap(True)
        combined_layout.addWidget(points_label)

        points_table = QtWidgets.QTableWidget(0, 2)
        points_table.horizontalHeader().setVisible(False)
        points_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        points_table.setMaximumHeight(140)

        points_header_row = QtWidgets.QHBoxLayout()
        points_header_row.setContentsMargins(4, 0, 4, 0)
        zeff_header_label = QtWidgets.QLabel(
            "<div style='font-size:10pt; font-weight:700;'>"
            "<i>Z<sub>eff</sub></i></div>"
            "<div style='font-size:7pt;'>(dimensionless)</div>"
        )
        zeff_header_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        zeff_header_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        rhoe_header_label = QtWidgets.QLabel(
            "<div style='font-size:10pt; font-weight:700;'>"
            "<i>\u03c1<sub>e</sub></i></div>"
            "<div style='font-size:7pt;'>"
            "(<i>e</i><sup>\u2212</sup>\u00b7<i>cm</i><sup>\u22123</sup>)"
            "</div>"
        )
        rhoe_header_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        rhoe_header_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        points_header_row.addWidget(zeff_header_label, 1)
        points_header_row.addWidget(rhoe_header_label, 1)
        combined_layout.addLayout(points_header_row)
        combined_layout.addWidget(points_table)

        points_buttons_row = QtWidgets.QHBoxLayout()
        export_points_button = QtWidgets.QPushButton("Export CSV...")
        clear_points_button = QtWidgets.QPushButton("Clear points")
        points_buttons_row.addWidget(export_points_button)
        points_buttons_row.addWidget(clear_points_button)
        combined_layout.addLayout(points_buttons_row)

        # Voxel x/y/z are kept internally (needed to read the values) but
        # are not shown to the user or exported -- not relevant to them.
        probed_points_data = []
        last_probed_voxel = {"value": None}

        def add_probed_point(x_index, y_index, z_index):
            raw_arrays = getattr(self, "geodect_3d_raw_arrays", {})
            if "zeff" not in raw_arrays or "rhoe" not in raw_arrays:
                return
            try:
                zeff_array = raw_arrays["zeff"]
                rhoe_array = raw_arrays["rhoe"]
                zeff_value = float(zeff_array[z_index, y_index, x_index])
                # The rho_e array stores DISPLAY-scaled values (divided
                # by rhoe_display_scale, e.g. 1e23) so the legend doesn't
                # show unreadable 24-digit numbers. Multiply back here so
                # probed points show the true physical rho_e value.
                rhoe_display_scale = float(
                    self.geodect_3d_display_settings.get(
                        "rhoe_display_scale", 1.0
                    )
                )
                rhoe_value = (
                    float(rhoe_array[z_index, y_index, x_index])
                    * rhoe_display_scale
                )
            except Exception:
                return

            probed_points_data.append(
                (x_index, y_index, z_index, zeff_value, rhoe_value)
            )
            row = points_table.rowCount()
            points_table.insertRow(row)
            zeff_item = QtWidgets.QTableWidgetItem(
                self._format_number(zeff_value)
            )
            zeff_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
            )
            rhoe_item = QtWidgets.QTableWidgetItem(
                self._format_number(rhoe_value)
            )
            rhoe_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
            )
            points_table.setItem(row, 0, zeff_item)
            points_table.setItem(row, 1, rhoe_item)
            points_table.scrollToBottom()
            # The scatter widget's axes are in DISPLAY-scaled rho_e units
            # (matching its full-range bounds), so the dot position needs
            # the unscaled value -- only the table/CSV show the true
            # physical rho_e value.
            scatter.add_point(
                zeff_value,
                rhoe_value / rhoe_display_scale if rhoe_display_scale else rhoe_value,
            )

        def clear_probed_points():
            probed_points_data.clear()
            points_table.setRowCount(0)
            scatter.clear_points()

        clear_points_button.clicked.connect(clear_probed_points)

        def export_probed_points():
            if not probed_points_data:
                self.volume_status_label.setText(
                    "No probed points to export yet."
                )
                return
            path, _filter = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export probed points", "geodect_probed_points.csv",
                "CSV files (*.csv)"
            )
            if not path:
                return
            try:
                with open(path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["Z_eff", "rho_e"])
                    for row_data in probed_points_data:
                        writer.writerow(row_data[3:])
                self.volume_status_label.setText(
                    "Probed points exported to {}".format(path)
                )
            except Exception as exc:
                self.volume_status_label.setText(
                    "Export failed: {}: {}".format(type(exc).__name__, exc)
                )

        export_points_button.clicked.connect(export_probed_points)

        _voxel_pattern = re.compile(
            r"Voxel:\s*x\s*:\s*(-?\d+)\s*y\s*:\s*(-?\d+)\s*z\s*:\s*(-?\d+)"
        )

        def poll_probe_tool():
            try:
                for widget in QtWidgets.QApplication.allWidgets():
                    if not widget.isVisible():
                        continue
                    text = None
                    if isinstance(widget, QtWidgets.QLabel):
                        text = widget.text()
                    elif hasattr(widget, "toPlainText"):
                        try:
                            text = widget.toPlainText()
                        except Exception:
                            text = None
                    if not text or "Voxel" not in text:
                        continue
                    match = _voxel_pattern.search(text)
                    if not match:
                        continue
                    voxel = (
                        int(match.group(1)),
                        int(match.group(2)),
                        int(match.group(3)),
                    )
                    if voxel != last_probed_voxel["value"]:
                        last_probed_voxel["value"] = voxel
                        add_probed_point(*voxel)
                    return
            except Exception:
                pass

        probe_timer = QtCore.QTimer(dock)
        probe_timer.setInterval(300)
        probe_timer.timeout.connect(poll_probe_tool)
        probe_timer.start()
        controls["probe_timer"] = probe_timer

        property_stack.addWidget(combined_container)

        def set_active_property(index):
            property_stack.setCurrentIndex(int(index))

        property_selector.currentIndexChanged.connect(set_active_property)
        property_selector.setCurrentIndex(0)
        property_stack.setCurrentIndex(0)

        # Dragonfly does not expose a stable public "selected dataset" signal
        # in every build. Poll visible item views and follow the selected
        # Geo_DECT row. This keeps the right panel synchronized when the user
        # clicks Z_eff or rho_e in Dragonfly's object list, while the combo box
        # remains a reliable manual fallback.
        def follow_dragonfly_selection():
            try:
                for view_widget in QtWidgets.QApplication.allWidgets():
                    if not isinstance(view_widget, QtWidgets.QAbstractItemView):
                        continue
                    if not view_widget.isVisible():
                        continue
                    selection_model = view_widget.selectionModel()
                    if selection_model is None:
                        continue
                    for index in selection_model.selectedIndexes():
                        text = str(index.data() or "").lower()
                        if "geo_dect" not in text and "z_eff" not in text and "rho_e" not in text:
                            continue
                        desired = None
                        # Check "combined" first: its own title contains
                        # BOTH "z_eff" and "rho_e" as substrings (e.g.
                        # "Geo_DECT — combined Z_eff / rho_e"), so checking
                        # z_eff/rho_e first would always misclassify it as
                        # the Z_eff page and immediately override any
                        # manual selection of "Combined" in the dropdown.
                        if "combined" in text:
                            desired = 2
                        elif "z_eff" in text or "zeff" in text:
                            desired = 0
                        elif "rho_e" in text or "rhoe" in text:
                            desired = 1
                        if desired is not None and property_selector.currentIndex() != desired:
                            property_selector.blockSignals(True)
                            property_selector.setCurrentIndex(desired)
                            property_selector.blockSignals(False)
                            property_stack.setCurrentIndex(desired)
                            return
            except Exception:
                pass

        selection_timer = QtCore.QTimer(dock)
        selection_timer.setInterval(250)
        selection_timer.timeout.connect(follow_dragonfly_selection)
        selection_timer.start()
        controls["selection_timer"] = selection_timer

        layout.addStretch(1)

        self._display_range_controls = controls

        dock.setWidget(body)
        self._display_range_dock = dock
        dock.destroyed.connect(self._on_display_range_dock_destroyed)

        add_dock = getattr(host, "addDockWidget", None)
        if callable(add_dock):
            add_dock(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        else:
            # Fallback for unusual Dragonfly window wrappers.
            dock.setFloating(True)
            dock.resize(330, 620)
        dock.show()
        dock.raise_()

    def _on_display_range_dock_destroyed(self, *_args):
        self._display_range_dock = None
        self._display_range_controls = {}

    def _create_and_apply_geodect_3d_luts(self):
        """Create preview-matched LUTs and apply them to the current layout."""
        from . import geodect_3d as geodect3d

        channels = getattr(self, "geodect_3d_channels", {})
        visuals = getattr(self, "geodect_3d_visuals", {})
        scene_layout = getattr(self, "geodect_3d_scene_layout", None)
        current_view = getattr(self, "geodect_3d_current_view", None)
        settings = getattr(self, "geodect_3d_display_settings", {})
        required_channels = ("zeff", "rhoe", "combined")

        missing = [
            key for key in required_channels
            if channels.get(key) is None
        ]
        if missing:
            raise RuntimeError(
                "Missing native channels for LUT application: "
                + ", ".join(missing)
            )

        missing_visuals = [
            key for key in required_channels
            if visuals.get(key) is None
        ]
        if missing_visuals:
            raise RuntimeError(
                "Missing VisualChannels for LUT application: "
                + ", ".join(missing_visuals)
            )

        if scene_layout is None:
            raise RuntimeError(
                "The scene layout used to create the VisualChannels "
                "is no longer available."
            )

        required_settings = (
            "zeff_minimum", "zeff_maximum",
            "rhoe_minimum", "rhoe_maximum",
        )
        missing_settings = [
            key for key in required_settings
            if key not in settings
        ]
        if missing_settings:
            raise RuntimeError(
                "Missing 3D display settings: "
                + ", ".join(missing_settings)
            )

        zeff_channel = channels["zeff"]
        rhoe_channel = channels["rhoe"]
        combined_channel = channels["combined"]

        palette_scale = 1.0 / 255.0
        zeff_palette = np.asarray(
            ZEFF_PALETTE_RGB255, dtype=np.float64
        ) * palette_scale
        rhoe_palette = np.asarray(
            RHOE_PALETTE_RGB255, dtype=np.float64
        ) * palette_scale

        zeff_lut = geodect3d.create_context_scalar_lut(
            "Geo_DECT Z_eff preview-matched",
            float(zeff_channel.getMinimumValue()),
            float(zeff_channel.getMaximumValue()),
            float(settings["zeff_minimum"]),
            float(settings["zeff_maximum"]),
            zeff_palette,
        )
        rhoe_lut = geodect3d.create_context_scalar_lut(
            "Geo_DECT rho_e preview-matched",
            float(rhoe_channel.getMinimumValue()),
            float(rhoe_channel.getMaximumValue()),
            float(settings["rhoe_minimum"]),
            float(settings["rhoe_maximum"]),
            rhoe_palette,
        )
        combined_lut = geodect3d.create_geodect_lut(
            name="Geo_DECT combined preview-matched",
        )

        luts = {
            "zeff": zeff_lut,
            "rhoe": rhoe_lut,
            "combined": combined_lut,
        }
        results = {}
        for key in ("zeff", "rhoe"):
            results[key] = geodect3d.apply_lut_to_channel_in_layout(
                channel=channels[key],
                lut=luts[key],
                scene_layout=scene_layout,
                visual_channel=visuals[key],
                view=current_view,
            )
            geodect3d.apply_nearest_interpolation(visuals[key])

        # Best-effort only, never allowed to affect the coloring above.
        # This applies to the whole layout (not per-channel -- the
        # signature is just (layoutFullName, bool)), hiding the native
        # scale bar/legend which can only ever show meaningless "0-4095"
        # codes once a channel is baked with display codes. Confirmed
        # working: needs an explicit 2D view refresh afterward.
        try:
            from OrsHelpers.viewLayoutHelper import DisplayLayoutHelper

            layout_name = scene_layout.getGenealogicalName()
            LayoutPropertiesHelper.setShowScaleBarFromGenealogicalName(
                layout_name, False
            )
            LayoutPropertiesHelper.setShowLegendFromGenealogicalName(
                layout_name, False
            )
            for a_view in DisplayLayoutHelper.getAll2DViewsInLayout(
                scene_layout
            ):
                a_view.refresh()
        except Exception:
            pass

        # "combined" needs apply_confirmed_2d_lut, NOT the older
        # apply_lut_to_channel_in_layout, which silently caps discrete
        # LUTs at 4096 entries (GEODECT_BINS=100 needs 10003).
        try:
            geodect3d.apply_confirmed_2d_lut(
                scene_layout, visuals["combined"], luts["combined"]
            )
            geodect3d.apply_nearest_interpolation(visuals["combined"])
            results["combined"] = "OK"
        except Exception as exc:
            results["combined"] = "{}: {}".format(
                type(exc).__name__, exc
            )


        failures = {
            key: result for key, result in results.items()
            if result != "OK"
        }
        if failures:
            details = "; ".join(
                f"{key}={result}"
                for key, result in sorted(failures.items())
            )
            raise RuntimeError(
                "One or more Dragonfly LUTs could not be attached: "
                + details
            )

        self.geodect_3d_luts = luts
        return results

    def _apply_confirmed_3d_coloring(self):
        """Color the Z_eff and rho_e 3D volumes.

        Confirmed working (2026-08): a continuous LUT applied via
        set3DLUTAsLUT + ColorRangeMode3D=WL + a refresh through
        DisplayLayoutHelper.getAll3DViewsInLayout -- none of which the
        generic 2D-oriented LUT application reliably triggers for the 3D
        renderer. Deliberately not touching OpacityMode3D (it reproducibly
        crashed Dragonfly's 3D renderer during testing).

        IMPORTANT: this must run AFTER every other refresh/window-level
        call in the 3D volume creation flow (the post-creation window-level
        loop and the _refresh_initial_zeff timers). Running it earlier was
        confirmed (2026-08) to have the coloring silently overwritten/
        reverted by those later generic refreshes, even though no error was
        reported. Only Z_eff and rho_e for now; "combined" needs its own
        bivariate approach, handled separately.
        """
        from . import geodect_3d as geodect3d

        channels = getattr(self, "geodect_3d_channels", {})
        visuals = getattr(self, "geodect_3d_visuals", {})
        scene_layout = getattr(self, "geodect_3d_scene_layout", None)
        settings = getattr(self, "geodect_3d_display_settings", {})

        if not channels or not visuals or scene_layout is None:
            self.volume_status_label.setText(
                "3D coloring: SKIPPED (missing channels/visuals/scene_layout: "
                "channels={}, visuals={}, scene_layout_is_none={})".format(
                    bool(channels), bool(visuals), scene_layout is None
                )
            )
            return

        self.volume_status_label.setText("3D coloring: STARTED")

        zeff_palette = np.asarray(
            ZEFF_PALETTE_RGB255, dtype=np.float64
        ) / 255.0
        rhoe_palette = np.asarray(
            RHOE_PALETTE_RGB255, dtype=np.float64
        ) / 255.0

        completed = []
        for key, palette in (("zeff", zeff_palette), ("rhoe", rhoe_palette)):
            if key not in channels or key not in visuals:
                continue
            try:
                selected_minimum = float(settings[f"{key}_minimum"])
                selected_maximum = float(settings[f"{key}_maximum"])
                continuous_lut = geodect3d.create_context_scalar_lut_continuous(
                    "Geo_DECT {} 3D".format(key),
                    float(channels[key].getMinimumValue()),
                    float(channels[key].getMaximumValue()),
                    selected_minimum,
                    selected_maximum,
                    palette,
                )
                geodect3d.apply_confirmed_3d_lut(
                    scene_layout, visuals[key], continuous_lut
                )
                completed.append(key)
                self.volume_status_label.setText(
                    "3D coloring: completed so far: {}".format(
                        ", ".join(completed)
                    )
                )
            except Exception as exc:
                self.volume_status_label.setText(
                    "3D coloring for {} failed: {}: {}".format(
                        key, type(exc).__name__, exc
                    )
                )

        # "combined" uses a discrete LUT (its scalar values are already
        # color-space codes, see encode_geodect_slice/create_geodect_lut) --
        # not a continuous property LUT like Z_eff/rho_e above. Confirmed
        # working with the same set3DLUTAsLUT/setColorRangeMode3D/
        # DisplayLayoutHelper refresh sequence.
        if "combined" in channels and "combined" in visuals:
            try:
                combined_lut = geodect3d.create_geodect_lut(
                    "Geo_DECT combined 3D"
                )
                geodect3d.apply_confirmed_3d_lut(
                    scene_layout, visuals["combined"], combined_lut
                )
                completed.append("combined")
                self.volume_status_label.setText(
                    "3D coloring: completed so far: {}".format(
                        ", ".join(completed)
                    )
                )
            except Exception as exc:
                self.volume_status_label.setText(
                    "3D coloring for combined failed: {}: {}".format(
                        type(exc).__name__, exc
                    )
                )

    def _initialize_geodect_scalar_collections(self, attempt):
        """Create ScalarValuesCollection objects after Dragonfly is ready.

        This method is intentionally driven by QTimer. Sleeping or polling in
        the GUI callback prevents Dragonfly from finishing creation of its
        VisualChannel objects. The native Channel data are never recreated or
        modified here.
        """
        channels = getattr(self, "geodect_3d_channels", {})
        if not channels:
            return

        state = getattr(self, "geodect_3d_visual_initialization", None)
        if not isinstance(state, dict):
            state = {"attempt": 0, "completed": False, "errors": {}}
            self.geodect_3d_visual_initialization = state

        state["attempt"] = int(attempt)
        errors = state.setdefault("errors", {})
        visuals = {}
        missing = []

        for key in ("zeff", "rhoe", "combined"):
            channel = channels.get(key)
            if channel is None:
                missing.append(key)
                errors[key] = "Channel reference is missing."
                continue
            try:
                visual = channel.getVisual()
            except Exception as exc:
                visual = None
                errors[key] = f"getVisual: {type(exc).__name__}: {exc}"
            if visual is None:
                missing.append(key)
            else:
                visuals[key] = visual
                errors.pop(key, None)

        self.geodect_3d_visuals = visuals

        if missing and attempt < 24:
            self.volume_status_label.setText(
                "Waiting for Dragonfly VisualChannels: "
                + ", ".join(missing)
                + f" (attempt {attempt}/24)."
            )
            QtCore.QTimer.singleShot(
                250,
                lambda next_attempt=attempt + 1:
                    self._initialize_geodect_scalar_collections(next_attempt),
            )
            return

        collections = {}
        for key, visual in visuals.items():
            collection = getattr(
                self, "geodect_3d_scalar_collections", {}
            ).get(key)
            try:
                if collection is None:
                    collection = visual.getScalarValuesCollection()
            except Exception as exc:
                errors[key] = (
                    f"getScalarValuesCollection: {type(exc).__name__}: {exc}"
                )

            if collection is None:
                try:
                    created = visual.createScalarValuesCollection()
                    # Depending on the Dragonfly build, the factory may return
                    # the collection or only attach it to the visual.
                    if created is not None:
                        collection = created
                    else:
                        collection = visual.getScalarValuesCollection()
                except Exception as exc:
                    errors[key] = (
                        "createScalarValuesCollection: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    collection = None

            if collection is not None:
                collections[key] = collection
                errors.pop(key, None)

            try:
                visual.set2DWindowLevelToShowFullRangeForAllViews()
                visual.set3DWindowLevelToShowFullRangeForAllViews()
                visual.refreshAllParentViewsIfVisible()
            except Exception as exc:
                errors.setdefault(
                    key,
                    f"visual refresh/full range: {type(exc).__name__}: {exc}",
                )

        self.geodect_3d_scalar_collections = collections
        state["completed"] = True

        missing_visuals = [
            key for key in ("zeff", "rhoe", "combined")
            if key not in visuals
        ]
        missing_collections = [
            key for key in ("zeff", "rhoe", "combined")
            if key in visuals and key not in collections
        ]

        if not missing_visuals and not missing_collections:
            try:
                lut_results = self._create_and_apply_geodect_3d_luts()
            except Exception as exc:
                errors["lut_application"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                state["completed"] = False
                self.volume_status_label.setText(
                    "VisualChannels are ready, but color LUT application "
                    f"failed: {errors['lut_application']}"
                )
                message = QtWidgets.QMessageBox(self)
                message.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                message.setWindowTitle(
                    "Geo_DECT — Color LUT application failed"
                )
                message.setText(str(exc))
                message.setDetailedText(traceback.format_exc())
                message.exec()
                return

            errors.pop("lut_application", None)
            state["completed"] = True
            state["lut_results"] = dict(lut_results)
            self.volume_status_label.setText(
                "Three colored native Dragonfly volumes are ready: "
                "Z_eff, rho_e and combined Geo_DECT."
            )
            QtWidgets.QMessageBox.information(
                self,
                "Geo_DECT — Colored 3D volumes ready",
                "Dragonfly created all three VisualChannels and Geo_DECT "
                "applied the preview-matched color LUTs.\n\n"
                "The Z_eff, rho_e and combined Geo_DECT volumes should now "
                "be visible and colored in the current layout.",
            )
        else:
            details = []
            if missing_visuals:
                details.append("VisualChannel missing: " + ", ".join(missing_visuals))
            if missing_collections:
                details.append(
                    "ScalarValuesCollection missing: "
                    + ", ".join(missing_collections)
                )
            for key, message in sorted(errors.items()):
                details.append(f"{key}: {message}")
            self.volume_status_label.setText(
                "Visual diagnostic initialization incomplete. "
                + " | ".join(details)
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Geo_DECT — Visual initialization incomplete",
                "The asynchronous initialization finished, but some objects "
                "are still unavailable.\n\n" + "\n".join(details),
            )