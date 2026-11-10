import json
import numpy as np
import os
import time
import traceback

_LUT_APPLICATION_STATE = {
    "in_progress": False,
    "last_call_time_by_channel": {},
}
_LUT_APPLICATION_MIN_INTERVAL_SECONDS = 1.0


# --- Persistence across Dragonfly project save/reload ----------------
#
# A native Channel survives a Dragonfly project save; the VisualChannel,
# the LUT and the chosen display range do NOT -- they are created fresh
# every time by Build()/Apply() and only ever live in this plugin's own
# Python-side state. setUserInfo/getUserInfo below are used to attach
# the chosen range directly to the Channel object itself, so it comes
# back when the project is reopened, and find_existing_geodect_channels()
# is used to find the channels themselves so they can be reconnected.

GEODECT_SETTINGS_TAG = "geodect_display_settings_json"
GEODECT_STATE_TAG = "geodect_display_state"  # "continuous" or "discrete"


def persist_display_settings(channel, settings, state):
    """Save the chosen display settings on the channel itself so they
    survive a Dragonfly project save/reload (VisualChannel/LUT do not --
    see find_existing_geodect_channels / the reconnect logic that reads
    this back)."""
    try:
        channel.setUserInfo(GEODECT_SETTINGS_TAG, json.dumps(settings))
        channel.setUserInfo(GEODECT_STATE_TAG, state)
    except Exception:
        pass


def read_persisted_display_settings(channel):
    """Read back whatever persist_display_settings() last stored on this
    channel. Returns (settings_dict_or_None, state_string)."""
    try:
        raw = channel.getUserInfo(GEODECT_SETTINGS_TAG)
        state = channel.getUserInfo(GEODECT_STATE_TAG) or "continuous"
        if not raw:
            return None, state
        return json.loads(raw), state
    except Exception:
        return None, "continuous"


def find_existing_geodect_channels(diagnostic_path=None):
    """Look for Geo_DECT channels already present in the scene --
    typically because a Dragonfly project that contained them was just
    reopened. Returns a dict with as many of 'zeff' / 'rhoe' / 'combined'
    as could be found (empty dict if none).

    Always writes a short diagnostic file (when diagnostic_path is
    given) describing exactly what was tried and what came back --
    unlike a plain empty-dict return, this makes it possible to tell
    "genuinely nothing to reconnect" apart from "the Dragonfly API call
    itself did not behave as expected here".
    """
    report_lines = []

    def log(message):
        report_lines.append(str(message))

    def write_report():
        if not diagnostic_path:
            return
        try:
            with open(diagnostic_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(report_lines))
        except Exception:
            pass

    try:
        from ORSModel import Channel
    except Exception as exc:
        log("from ORSModel import Channel FAILED: {}: {}".format(
            type(exc).__name__, exc
        ))
        write_report()
        return {}, {}

    try:
        from ORSModel import VisualChannel
    except Exception as exc:
        VisualChannel = None
        log(
            "from ORSModel import VisualChannel FAILED (existing "
            "VisualChannels will not be reused): {}: {}".format(
                type(exc).__name__, exc
            )
        )

    # Try every calling convention seen across Dragonfly/ORS versions.
    # getAllObjectsOfClass("Channel") succeeding but returning 0 objects
    # (confirmed 2026-09 on Dragonfly 2025.1) means the channels Build
    # creates are registered under a different internal class name, not
    # literally "Channel" -- so getAllRepresentableObjects(), which does
    # not filter by class at all, is tried first.
    all_channels = None
    for attempt_name, attempt in (
        ("Channel.getAllRepresentableObjects()",
         lambda: Channel.getAllRepresentableObjects()),
        ('Channel.getAllObjectsOfClass("Channel")',
         lambda: Channel.getAllObjectsOfClass("Channel")),
        ("Channel.getAllObjectsOfClass()",
         lambda: Channel.getAllObjectsOfClass()),
        ('Channel.getAllObjectsOfClass("ORSModel.ors.Channel")',
         lambda: Channel.getAllObjectsOfClass("ORSModel.ors.Channel")),
    ):
        try:
            result = attempt()
            count = len(result)
            log("{} -> OK, {} object(s).".format(attempt_name, count))
            if count:
                all_channels = result
                break
        except Exception as exc:
            log("{} -> FAILED: {}: {}".format(
                attempt_name, type(exc).__name__, exc
            ))

    if not all_channels:
        log("No calling convention returned any objects.")
        write_report()
        return {}, {}

    found = {}
    found_visuals = {}
    all_titles = []
    filtered_out_count = 0
    for channel in all_channels:
        try:
            title = channel.getTitle() or ""
        except Exception:
            # getAllRepresentableObjects() returns every representable
            # object in the scene, not just channels (layouts, ROIs,
            # meshes...) -- most of those simply don't expose
            # getTitle() the same way, that is expected and not an
            # error worth logging for every single one of them.
            continue

        # Confirmed 2026-09: a VisualChannel attached to one of these
        # channels SURVIVES a Dragonfly project save/reload and reports
        # the same title as the channel it wraps -- it is not recreated
        # from scratch here as previously assumed. Creating a brand new
        # one instead of reusing this one leaves two VisualChannels
        # attached to the same Channel: the old one (still the one
        # actually shown/toggleable in the Properties panel, stuck
        # grey) and the new one this plugin colors but that never
        # appears anywhere. So the existing VisualChannel is captured
        # here to be reused, not discarded.
        if VisualChannel is not None and isinstance(channel, VisualChannel):
            key = None
            if title.startswith("Geo_DECT — Z_eff"):
                key = "zeff"
            elif title.startswith("Geo_DECT — rho_e"):
                key = "rhoe"
            elif title == "Geo_DECT — combined Z_eff / rho_e":
                key = "combined"
            if key:
                found_visuals[key] = channel
            continue

        if not isinstance(channel, Channel):
            filtered_out_count += 1
            continue

        all_titles.append(title)
        if title.startswith("Geo_DECT — Z_eff"):
            found["zeff"] = channel
        elif title.startswith("Geo_DECT — rho_e"):
            found["rhoe"] = channel
        elif title == "Geo_DECT — combined Z_eff / rho_e":
            found["combined"] = channel

    log("Titles seen ({} total, {} filtered out as non-Channel, "
        "non-Geo_DECT ones truncated):".format(
            len(all_titles), filtered_out_count
        ))
    for title in all_titles:
        if title.startswith("Geo_DECT") or len(all_titles) <= 60:
            log("  - {!r}".format(title))
    log("Matched channels: {}".format(sorted(found.keys())))
    log("Matched existing VisualChannels: {}".format(
        sorted(found_visuals.keys())
    ))
    write_report()

    return found, found_visuals


def find_visual_channels_in_layout(scene_layout, diagnostic_path=None):
    """Look for Geo_DECT VisualChannels already attached to scene_layout
    itself, not just anywhere "in the world".

    find_existing_geodect_channels() searches globally via
    getAllRepresentableObjects(), which is confirmed (2026-09) to find
    the Channel objects but NOT any VisualChannel already nested inside
    a layout's own child hierarchy -- a VisualChannel attached to a
    view is apparently not surfaced by that global sweep the same way.
    This instead walks scene_layout's own children directly, which is
    where Dragonfly keeps whatever is actually shown in a view.

    Returns a dict with as many of 'zeff' / 'rhoe' / 'combined' as
    could be found (empty dict if none), and always writes a
    diagnostic file describing what was tried, same spirit as
    find_existing_geodect_channels().
    """
    report_lines = []

    def log(message):
        report_lines.append(str(message))

    def write_report():
        if not diagnostic_path:
            return
        try:
            with open(diagnostic_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(report_lines))
        except Exception:
            pass

    if scene_layout is None:
        log("scene_layout is None -- nothing to search.")
        write_report()
        return {}

    all_nodes = None
    for attempt_name, attempt in (
        ('scene_layout.getAllChildrenOfClass("VisualChannel")',
         lambda: scene_layout.getAllChildrenOfClass("VisualChannel")),
        ('scene_layout.getAllChildrenOfClass("Channel")',
         lambda: scene_layout.getAllChildrenOfClass("Channel")),
        ("scene_layout.getFlattenedChildrenHierarchy()",
         lambda: scene_layout.getFlattenedChildrenHierarchy()),
    ):
        try:
            result = attempt()
            count = len(result)
            log("{} -> OK, {} object(s).".format(attempt_name, count))
            if count:
                all_nodes = result
                break
        except Exception as exc:
            log("{} -> FAILED: {}: {}".format(
                attempt_name, type(exc).__name__, exc
            ))

    if not all_nodes:
        log("No calling convention returned any objects.")
        write_report()
        return {}

    found = {}
    all_titles = []
    for node in all_nodes:
        try:
            title = node.getTitle() or ""
        except Exception:
            continue
        all_titles.append((title, type(node).__name__))
        if title.startswith("Geo_DECT — Z_eff"):
            found["zeff"] = node
        elif title.startswith("Geo_DECT — rho_e"):
            found["rhoe"] = node
        elif title == "Geo_DECT — combined Z_eff / rho_e":
            found["combined"] = node

    log("Titles seen in scene_layout hierarchy ({} total):".format(
        len(all_titles)
    ))
    for title, type_name in all_titles:
        if title.startswith("Geo_DECT") or len(all_titles) <= 60:
            log("  - {!r} (type: {})".format(title, type_name))
    log("Matched: {}".format(sorted(found.keys())))
    write_report()

    return found


from .geodect_visualization import (
    ABOVE_RGB255,
    BACKGROUND_RGB255,
    BELOW_RGB255,
    GEODECT_A_MAX,
    GEODECT_A_MIN,
    GEODECT_B_MAX,
    GEODECT_B_MIN,
    GEODECT_LIGHTNESS,
    RHOE_PALETTE_RGB255,
    ZEFF_PALETTE_RGB255,
    lab_to_rgb,
)

def _channel_class():
    """Import only the native Channel class when channel allocation starts."""
    from ORSModel import Channel
    return Channel


def _color_class():
    """Import Color only when a LUT is actually being built."""
    from ORSModel import Color
    return Color

def _lut_logic_class():
    """
    Locate the LUT creation helper used by the installed Dragonfly version.

    The returned adapter always exposes:

        createLUT(name)
        getLUTUUID(lut)

    Returns None only when no usable LUT implementation can be found.
    """
    import importlib
    import sys

    def make_adapter(provider):
        """Normalize the different LUT helper method names."""

        create_method = None
        uuid_method = None

        for method_name in (
            "createLUT",
            "createLut",
            "createLookupTable",
        ):
            candidate = getattr(provider, method_name, None)
            if callable(candidate):
                create_method = candidate
                break

        if create_method is None:
            return None

        for method_name in (
            "getLUTUUID",
            "getLutUUID",
            "getLUTUuid",
            "getUUID",
        ):
            candidate = getattr(provider, method_name, None)
            if callable(candidate):
                uuid_method = candidate
                break

        class LUTLogicAdapter:
            @staticmethod
            def createLUT(name):
                try:
                    return create_method(name)
                except TypeError:
                    # Some implementations create an unnamed LUT.
                    lut = create_method()

                    # Set the name only when the returned class supports it.
                    for setter_name in (
                        "setTitle",
                        "setName",
                    ):
                        setter = getattr(lut, setter_name, None)
                        if callable(setter):
                            try:
                                setter(str(name))
                                break
                            except Exception:
                                pass

                    return lut

            @staticmethod
            def getLUTUUID(lut):
                if uuid_method is not None:
                    try:
                        value = uuid_method(lut)
                        if value:
                            return str(value)
                    except Exception:
                        pass

                # Fall back to methods exposed directly by the LUT object.
                for method_name in (
                    "getLUTUUID",
                    "getLutUUID",
                    "getUUID",
                    "getGUID",
                    "getUid",
                ):
                    method = getattr(lut, method_name, None)

                    if callable(method):
                        try:
                            value = method()
                            if value:
                                return str(value)
                        except Exception:
                            pass

                return None

        return LUTLogicAdapter

    def inspect_container(container):
        """Find a LUT provider inside a module or class."""

        if container is None:
            return None

        # The module itself may expose createLUT.
        adapter = make_adapter(container)
        if adapter is not None:
            return adapter

        # Names used by different Dragonfly releases.
        preferred_names = (
            "LUTHelper",
            "LutHelper",
            "OrsLUTLogic",
            "OrsLutLogic",
            "LUTLogic",
            "LutLogic",
        )

        for attribute_name in preferred_names:
            provider = getattr(container, attribute_name, None)
            adapter = make_adapter(provider)

            if adapter is not None:
                return adapter

        # Last resort: inspect all objects exported by the module.
        for attribute_name in dir(container):
            if attribute_name.startswith("_"):
                continue

            try:
                provider = getattr(container, attribute_name)
            except Exception:
                continue

            adapter = make_adapter(provider)

            if adapter is not None:
                return adapter

        return None

    module_candidates = (
        # Dragonfly 2025.1 helper module.
        "OrsHelpers.luthelper",

        # Historical Dragonfly modules.
        "ORSServiceClass.ORSWidget.orslutwidget.orslutlogic",
        "ORSServiceClass.ORSWidget.orslutwidget.orsLUTLogic",

        # Additional paths used by some installations.
        "OrsPythonPlugins.OrsWindowLeveling.orslutlogic",
        "OrsPythonPlugins.OrsWindowLeveling.orsLUTLogic",
        "orslut.orslutlogic",
        "orslut.orsLUTLogic",
    )

    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        adapter = inspect_container(module)

        if adapter is not None:
            return adapter

    # Inspect LUT-related modules that Dragonfly has already loaded.
    for module_name, module in tuple(sys.modules.items()):
        if module is None or "lut" not in module_name.lower():
            continue

        try:
            adapter = inspect_container(module)
        except Exception:
            continue

        if adapter is not None:
            return adapter

    # Final fallback: create a native ORS LookupTable directly.
    try:
        from ORSModel import LookupTable
    except Exception:
        return None

    class NativeLookupTableAdapter:
        @staticmethod
        def createLUT(name):
            lut = LookupTable()

            for setter_name in (
                "setTitle",
                "setName",
            ):
                setter = getattr(lut, setter_name, None)

                if callable(setter):
                    try:
                        setter(str(name))
                        break
                    except Exception:
                        pass

            return lut

        @staticmethod
        def getLUTUUID(lut):
            for method_name in (
                "getLUTUUID",
                "getLutUUID",
                "getUUID",
                "getGUID",
                "getUid",
            ):
                method = getattr(lut, method_name, None)

                if callable(method):
                    try:
                        value = method()
                        if value:
                            return str(value)
                    except Exception:
                        pass

            return None

    return NativeLookupTableAdapter

def _print_filtered_methods(label, obj):
    """Print methods and attributes related to LUT and scalar values."""
    print("\n===== {} =====".format(label))

    if obj is None:
        print("Object is None")
        return

    print("Object:", repr(obj))
    print("Python type:", type(obj))

    try:
        attribute_names = dir(obj)
    except Exception as exc:
        print(
            "Unable to inspect object: {}: {}".format(
                type(exc).__name__,
                exc,
            )
        )
        return

    keywords = (
        "scalar",
        "tag",
        "lut",
        "lookup",
        "type",
        "value",
        "collection",
    )

    matching_names = []

    for name in attribute_names:
        lower_name = name.lower()

        if any(keyword in lower_name for keyword in keywords):
            matching_names.append(name)

    if not matching_names:
        print("No matching attributes found.")
        return

    for name in sorted(matching_names):
        try:
            value = getattr(obj, name)
        except Exception as exc:
            print(
                "{} -> ERROR: {}: {}".format(
                    name,
                    type(exc).__name__,
                    exc,
                )
            )
            continue

        if callable(value):
            print("{} -> method".format(name))
        else:
            try:
                print("{} -> {!r}".format(name, value))
            except Exception:
                print("{} -> value not printable".format(name))


def _view_api():
    """Load view/layout classes only when automatic LUT attachment is tried."""
    try:
        from ORSModel import VisualChannel
        from OrsHelpers.layoutpropertieshelper import LayoutPropertiesHelper
        from OrsHelpers.layoutHelper import LayoutHelper
        from OrsLibraries.workingcontext import WorkingContext
    except ImportError:
        return None

    return {
        "VisualChannel": VisualChannel,
        "LayoutPropertiesHelper": LayoutPropertiesHelper,
        "LayoutHelper": LayoutHelper,
        "WorkingContext": WorkingContext,
    }


GEODECT_BINS = 100
# 100*100+3 = 10003 total LUT entries -- intermediate value between the
# stable 63 and the 255 that caused a freeze (2026-08). Still above the
# 4096 cap of the old apply_lut_to_channel_in_layout path, so "combined"
# must use apply_confirmed_2d_lut/apply_confirmed_3d_lut, called
# SEPARATELY (never both in the same function/button click -- that
# combination froze Dragonfly at n_bins=255).
GEODECT_BACKGROUND_CODE = 0
GEODECT_BELOW_CODE = 1
GEODECT_ABOVE_CODE = 2
GEODECT_FIRST_COLOR_CODE = 3


def apply_confirmed_3d_lut(scene_layout, visual_channel, lut):
    """
    Apply and display a LUT on a 3D volume.

    This uses the exact sequence confirmed working interactively in the
    Dragonfly Python console (2026-08), after extensive testing showed
    that the LUT-assignment paths used elsewhere in this file (set3DLUTUUID,
    the generic refreshAllParentViews()/view.refresh() calls) update an
    internal "proxy" LUT object and/or the object icon, but do NOT cause
    the actual 3D renderer to redraw with the new colors:

    1. LayoutPropertiesHelper.set3DLUTAsLUT(scene_layout, visual_channel,
       lut) -- passing the VisualChannel and the LUT object directly (NOT
       set3DLUTUUID, which was confirmed to only update the range/window-
       level mapping, not the actual displayed colors).
    2. LayoutPropertiesHelper.setColorRangeMode3D(..., CXVCOLORRANGE_MODE_WL)
       -- without this the 3D volume ignores the LUT and shows raw/native
       data-range grayscale.
    3. Refreshing every 3D view via
       DisplayLayoutHelper.getAll3DViewsInLayout(scene_layout) -- this is
       what the native Window Leveling panel itself does
       (OrsWindowLeveling.py) and is different from the generic
       refreshAllParentViews()/view.refresh() calls used elsewhere in this
       file, which do not reliably trigger a 3D redraw.

    IMPORTANT: OpacityMode3D is deliberately left untouched here. Forcing
    it (e.g. to match the 2D opacity mode) reproducibly crashed Dragonfly's
    3D renderer during testing, even on small volumes. Do not add that back
    without first testing extensively on a small/cropped dataset.

    The supplied LUT should be a continuous LUT (setIsDiscrete(False)) --
    this is what was used when the fix was confirmed working. A discrete
    LUT has not been re-tested with this exact call sequence.
    """
    from OrsHelpers.layoutpropertieshelper import LayoutPropertiesHelper
    from OrsHelpers.viewLayoutHelper import DisplayLayoutHelper
    from COMWrapper.ORS_def import CxvColorRange_Mode

    LayoutPropertiesHelper.set3DLUTAsLUT(
        scene_layout, visual_channel, lut, logging=False
    )
    LayoutPropertiesHelper.setColorRangeMode3D(
        scene_layout, visual_channel, CxvColorRange_Mode.CXVCOLORRANGE_MODE_WL
    )

    all_3d_views = DisplayLayoutHelper.getAll3DViewsInLayout(scene_layout)
    for a_view in all_3d_views:
        a_view.refresh()


def apply_nearest_interpolation(visual_channel):
    """
    Disable smoothing/interpolation on a VisualChannel, for both 2D and
    3D, so that discrete/categorical data (like the "combined" codes)
    are not blended between neighbouring voxels.

    Confirmed working interactively in the Dragonfly console (2026-08),
    for all three channels (Z_eff, rho_e, combined), verified via the
    "Interpolation" dropdown in the native Properties panel:
    - 2D: VisualChannel.setFilteringMode(CxvFiltering_Mode.CXV_FILTER_POINT)
      -- the earlier setUseTriCubicFilteringIn2D(False) attempt did NOT
      work for 2D; setFilteringMode is the real control behind the "2D
      settings -> Interpolation" dropdown.
    - 3D: VisualChannel.setUsePointFilteringIn3D(True) -- this one did
      work as originally guessed, confirmed against the "3D settings ->
      Interpolation" dropdown.
    """
    from COMWrapper.ORS_def import CxvFiltering_Mode

    try:
        visual_channel.setFilteringMode(
            int(CxvFiltering_Mode.CXV_FILTER_POINT)
        )
    except Exception:
        pass

    set_point_3d = getattr(visual_channel, "setUsePointFilteringIn3D", None)
    if callable(set_point_3d):
        try:
            set_point_3d(True)
        except Exception:
            pass


def apply_confirmed_2d_lut(scene_layout, visual_channel, lut):
    """
    Apply and display a LUT on a 2D view.

    2D equivalent of apply_confirmed_3d_lut(), for the same reason: the
    older LUT-assignment path in this file (apply_lut_to_channel_in_layout,
    with its many fallback method attempts) was confirmed (2026-08) to
    silently cap discrete LUTs at 4096 entries when applied through
    Dragonfly's internal 2D LUT clone -- enough for Z_eff/rho_e alone, but
    far too coarse for the combined Z_eff/rho_e colour space (which needs
    up to ~65000 entries for a smooth 255x255 bivariate palette). A raw
    LookupTable() combined with set2DLUTAsLUT was confirmed to preserve
    the full requested table size (tested up to 65028 entries).

    Mirrors OrsWindowLeveling.py's own setColorRangeModeInView, which
    (for the 2D case) refreshes via
    DisplayLayoutHelper.getAll2DViewsInLayout(layout) after changing the
    colour range mode -- not the generic refreshAllParentViews()/
    view.refresh() calls used elsewhere in this file.
    """
    from OrsHelpers.layoutpropertieshelper import LayoutPropertiesHelper
    from OrsHelpers.viewLayoutHelper import DisplayLayoutHelper
    from COMWrapper.ORS_def import CxvColorRange_Mode

    LayoutPropertiesHelper.set2DLUTAsLUT(
        scene_layout, visual_channel, lut, logging=False
    )
    LayoutPropertiesHelper.setColorRangeMode2D(
        scene_layout, visual_channel, CxvColorRange_Mode.CXVCOLORRANGE_MODE_WL
    )

    # Without this, transparent LUT endpoints can render as an opaque
    # colour instead of actually being transparent (see the same call in
    # apply_lut_to_channel_in_layout for context).
    try:
        layout_name = scene_layout.getGenealogicalName()
        channel = visual_channel.getModel()
        LayoutPropertiesHelper.setUse2DLUTAlphaFromGenealogicalName(
            layout_name, channel, True
        )
    except Exception:
        pass

    all_2d_views = DisplayLayoutHelper.getAll2DViewsInLayout(scene_layout)
    for a_view in all_2d_views:
        a_view.refresh()


def create_context_scalar_lut_continuous(
    name,
    data_minimum,
    data_maximum,
    selected_minimum,
    selected_maximum,
    palette,
    table_size=256,
):
    """
    Continuous-LUT variant of create_context_scalar_lut, for use with
    apply_confirmed_3d_lut(). Builds the LUT directly with LookupTable()
    (not through the createLUT()/LUTLogic factory) and sets
    setIsDiscrete(False), matching what was confirmed working for 3D.

    Values outside the selected range are always fully transparent.
    """
    from ORSModel import LookupTable, Color

    table_size = max(int(table_size), 2)

    selected_minimum = float(selected_minimum)
    selected_maximum = float(selected_maximum)
    if selected_maximum <= selected_minimum:
        selected_maximum = selected_minimum + 1.0e-12

    lut = LookupTable()
    lut.setTitle(name)
    lut.setIsDiscrete(False)
    lut.setTableSize(table_size)

    interior_denominator = max(table_size - 4, 1)
    for index in range(table_size):
        alpha = 1.0
        if index in (0, 1) or index == table_size - 1:
            rgb = np.asarray((0, 0, 0), dtype=np.float64)
            alpha = 0.0
        else:
            normalized = np.clip(
                (index - 2) / float(interior_denominator), 0.0, 1.0
            )
            rgb = _interpolate_palette(normalized, palette)

        lut.setColorAtIndex(
            index,
            Color(float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha)),
        )

    lut.setIsRepresentable(True)
    lut.publish()
    return lut


GEODECT_MAX_CODE = (
    GEODECT_FIRST_COLOR_CODE + GEODECT_BINS * GEODECT_BINS - 1
)


def set_geodect_bins(n_bins):
    """
    Change GEODECT_BINS and keep GEODECT_MAX_CODE in sync in one call.

    Always use this instead of setting GEODECT_BINS directly -- a
    mismatch between the two (e.g. changing GEODECT_BINS without
    recomputing GEODECT_MAX_CODE) caused a full outage in 2026-08 that
    looked like a Dragonfly bug but was really just this desync.
    """
    global GEODECT_BINS, GEODECT_MAX_CODE
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")
    GEODECT_BINS = n_bins
    GEODECT_MAX_CODE = GEODECT_FIRST_COLOR_CODE + n_bins * n_bins - 1


REGISTERED_LUT_UUIDS = {
    "Blue": "211e9beeeef311e6894f448a5b87686a",
    "Green": "ecf0bc5ceef211e68270448a5b87686a",
    "Cyan": "7a45dd5eeef311e698d1448a5b87686a",

    "Red2": "c87ef590eef211e6a788448a5b87686a",
    "Yellow": "3d4a949aeef311e69c76448a5b87686a",
    "Magenta2": "5a2ae7feeef311e6a45d448a5b87686a",

    "Rainbow": "be71f848eef711e6aab4448a5b87686a",
    "Spectrum": "5fe7a8e2eefb11e68616448a5b87686a",
    "Ice": "0fd8c8b4eefb11e68485448a5b87686a",
}


def apply_registered_lut_uuid(channel, lut_uuid):
    """Assign a registered LUT to both the 2D and 3D current layouts."""
    api = _view_api()

    if api is None:
        return "API"
    if channel is None:
        return "CHANNEL"
    if not lut_uuid:
        return "LUT UUID"

    WorkingContext = api["WorkingContext"]
    LayoutHelper = api["LayoutHelper"]
    LayoutPropertiesHelper = api["LayoutPropertiesHelper"]

    try:
        view = WorkingContext.getCurrentView(None)
    except Exception as exc:
        return f"VIEW ERROR: {type(exc).__name__}: {exc}"

    if view is None:
        return "VIEW"

    try:
        scene_layout = LayoutHelper.getSceneLayoutOfView(view)
    except Exception as exc:
        return f"LAYOUT ERROR: {type(exc).__name__}: {exc}"

    if scene_layout is None:
        return "LAYOUT"

    layout_full_name = None

    for method_name in (
        "getFullName",
        "getGenealogicalName",
        "getFullNameWithPath",
    ):
        method = getattr(scene_layout, method_name, None)

        if not callable(method):
            continue

        try:
            value = method()

            if value:
                layout_full_name = str(value)
                break
        except Exception:
            pass

    if not layout_full_name:
        return "LAYOUT NAME ERROR"

    try:
        LayoutPropertiesHelper.set3DLUTUUIDFromGenealogicalName(
            layoutFullName=layout_full_name,
            anObject=channel,
            lutUUID=str(lut_uuid),
            aScalarValueTypeTag=""
        )

        LayoutPropertiesHelper.set2DLUTUUIDFromGenealogicalName(
            layoutFullName=layout_full_name,
            anObject=channel,
            lutUUID=str(lut_uuid),
            aScalarValueTypeTag=""
        )

    except Exception as exc:
        return (
            f"LUT ASSIGNMENT ERROR: {type(exc).__name__}: {exc}; "
            f"layout={layout_full_name!r}; "
            f"lut_uuid={lut_uuid!r}"
        )

    try:
        visual = channel.getVisual()

        if visual is not None:
            visual.refreshAllParentViews()
    except Exception:
        pass

    try:
        view.refresh()
    except Exception:
        pass

    return (
        "OK; "
        f"layout={layout_full_name!r}; "
        f"lut_uuid={str(lut_uuid)!r}"
    )

def estimate_output_bytes(shape_zyx):
    """Estimate bytes used by two float32 channels and one uint16 crossplot channel."""
    voxel_count = int(np.prod(shape_zyx, dtype=np.int64))
    return voxel_count * (4 + 4 + 2)


def format_bytes(byte_count):
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0


def create_native_channel(volume_zyx, dtype, title, spacing_xyz_m):
    """Create and validate an unpublished Dragonfly Channel from a NumPy volume.

    Dragonfly 2025.1 returns a new Channel from ``fromNDArray``. The returned
    object must be retained; the original empty Channel remains uninitialized.
    """
    Channel = _channel_class()

    requested_dtype = np.dtype(dtype)
    supported = (np.dtype(np.float32), np.dtype(np.uint16))
    if requested_dtype not in supported:
        raise TypeError(
            f"Unsupported Dragonfly channel dtype: {requested_dtype}"
        )

    array = np.asarray(volume_zyx, dtype=requested_dtype)
    if array.ndim != 3:
        raise ValueError(
            f"{title}: expected a 3D (z, y, x) array, got {array.shape}."
        )
    if any(int(size) <= 0 for size in array.shape):
        raise ValueError(f"{title}: volume dimensions must be positive.")

    array = np.ascontiguousarray(array)

    if requested_dtype == np.dtype(np.float32) and not np.all(np.isfinite(array)):
        array = np.nan_to_num(
            array,
            copy=True,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    # CRITICAL: fromNDArray returns the initialized Channel object.
    channel = Channel().fromNDArray(array)

    if channel is None:
        raise RuntimeError(
            f"{title}: Dragonfly fromNDArray returned None."
        )
    if not channel.getIsDataInitialized():
        raise RuntimeError(
            f"{title}: Dragonfly created an uninitialized channel."
        )

    expected_shape = tuple(int(value) for value in array.shape)
    native_shape = (
        int(channel.getZSize()),
        int(channel.getYSize()),
        int(channel.getXSize()),
    )
    if native_shape != expected_shape:
        raise RuntimeError(
            f"{title}: Dragonfly shape mismatch; source={expected_shape}, "
            f"native={native_shape}."
        )

    if requested_dtype == np.dtype(np.float32):
        if not channel.getIsDataTypeFloat():
            raise RuntimeError(
                f"{title}: Dragonfly did not preserve float32 data."
            )
    elif not channel.getIsDataTypeUnsignedShort():
        raise RuntimeError(
            f"{title}: Dragonfly did not preserve uint16 data."
        )

    readback = np.asarray(channel.getNDArray(0))
    if readback.shape != array.shape:
        raise RuntimeError(
            f"{title}: readback shape mismatch; source={array.shape}, "
            f"readback={readback.shape}."
        )

    source_min = float(np.min(array))
    source_max = float(np.max(array))
    native_min = float(channel.getMinimumValue())
    native_max = float(channel.getMaximumValue())
    tolerance = max(
        1.0e-6,
        abs(source_min) * 1.0e-5,
        abs(source_max) * 1.0e-5,
    )
    if abs(native_min - source_min) > tolerance:
        raise RuntimeError(
            f"{title}: minimum mismatch; source={source_min}, "
            f"native={native_min}."
        )
    if abs(native_max - source_max) > tolerance:
        raise RuntimeError(
            f"{title}: maximum mismatch; source={source_max}, "
            f"native={native_max}."
        )

    try:
        x_spacing, y_spacing, z_spacing = (
            float(value) for value in spacing_xyz_m
        )
    except Exception as exc:
        raise ValueError(
            "spacing_xyz_m must contain x, y and z spacing values."
        ) from exc

    if not all(
        np.isfinite(value) and value > 0.0
        for value in (x_spacing, y_spacing, z_spacing)
    ):
        raise ValueError(
            f"{title}: voxel spacings must be finite and positive."
        )

    channel.setXSpacing(x_spacing)
    channel.setYSpacing(y_spacing)
    channel.setZSpacing(z_spacing)
    channel.setTitle(str(title))
    return channel


def encode_geodect_slice(
    zeff,
    rhoe,
    valid_mask,
    zeff_minimum,
    zeff_maximum,
    rhoe_minimum,
    rhoe_maximum,
):
    """
    Encode one slice into a discrete Geo_DECT index volume.

    0 = background
    1 = below selected range (dark gray)
    2 = above selected range (white)
    3... = full 2D CIE L*a*b* color space, rescaled to selected ranges
    """
    zeff = np.asarray(zeff)
    rhoe = np.asarray(rhoe)
    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(zeff)
        & np.isfinite(rhoe)
        & (zeff > 0.0)
        & (rhoe > 0.0)
    )

    codes = np.zeros(zeff.shape, dtype=np.uint16)

    below = valid & (
        (zeff < zeff_minimum)
        | (rhoe < rhoe_minimum)
    )
    above = valid & (~below) & (
        (zeff > zeff_maximum)
        | (rhoe > rhoe_maximum)
    )
    inside = valid & (~below) & (~above)

    codes[below] = GEODECT_BELOW_CODE
    codes[above] = GEODECT_ABOVE_CODE

    if np.any(inside):
        zeff_n = np.clip(
            (zeff[inside] - zeff_minimum)
            / (zeff_maximum - zeff_minimum),
            0.0,
            1.0
        )
        rhoe_n = np.clip(
            (rhoe[inside] - rhoe_minimum)
            / (rhoe_maximum - rhoe_minimum),
            0.0,
            1.0
        )

        zeff_bin = np.minimum(
            (zeff_n * (GEODECT_BINS - 1)).astype(np.uint16),
            GEODECT_BINS - 1
        )
        rhoe_bin = np.minimum(
            (rhoe_n * (GEODECT_BINS - 1)).astype(np.uint16),
            GEODECT_BINS - 1
        )

        codes[inside] = (
            GEODECT_FIRST_COLOR_CODE
            + rhoe_bin * GEODECT_BINS
            + zeff_bin
        )

    return codes


def _set_color(lut, index, rgb, alpha=1.0):
    Color = _color_class()
    r, g, b = [float(value) for value in rgb]
    lut.setColorAtIndex(
        int(index),
        Color(r, g, b, float(alpha))
    )


def _finalize_lut(lut):
    """Commit and publish a custom LUT before its UUID is attached to a view."""
    if lut is None:
        raise ValueError("Cannot finalize a null LUT.")

    dirty_method = getattr(lut, "setDataDirty", None)
    if callable(dirty_method):
        dirty_method()

    publish_method = getattr(lut, "publish", None)
    if callable(publish_method):
        publish_method()

    return lut


SCALAR_BACKGROUND_CODE = 0
SCALAR_BELOW_CODE = 1
SCALAR_ABOVE_CODE = 2
# SCALAR_ABOVE_CODE is duplicated onto index 3 too (see
# create_scalar_context_lut) -- confirmed 2026-08 that index 2 alone was
# unreliable, same class of quirk as "index 0 is a dead slot". The
# gradient therefore starts one index later than before.
SCALAR_FIRST_COLOR_CODE = 4
SCALAR_BINS = 4092
# 4092 + 4 reserved codes (0,1,2,3) = 4096 total entries -- stays at the
# old, already-confirmed-safe apply_lut_to_channel_in_layout cap, so this
# path never needs apply_confirmed_2d_lut/apply_confirmed_3d_lut and
# never risks the freeze those caused when combined with "combined".


def encode_scalar_slice(values, valid_mask, selected_minimum, selected_maximum):
    """
    Encode one slice of a SINGLE property (Z_eff or rho_e) into a
    discrete index array, the same "baked codes + matching discrete LUT"
    approach already used for "combined" -- instead of relying on
    Dragonfly's live Window/Level colour mapping, which was confirmed
    (2026-08) to have real bugs (the "Geo_DECT display range" sliders not
    actually updating the displayed Window/Level; LUT index 0 never being
    rendered at all).

    0 = background (no data)
    1 = below the selected range (always fully transparent)
    2 = above the selected range (always fully transparent)
    3... = the property's own colour palette, rescaled to the selected
           range
    """
    values = np.asarray(values)
    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(values)
        & (values > 0.0)
    )

    codes = np.zeros(values.shape, dtype=np.uint16)

    below = valid & (values < selected_minimum)
    above = valid & (~below) & (values > selected_maximum)
    inside = valid & (~below) & (~above)

    codes[below] = SCALAR_BELOW_CODE
    codes[above] = SCALAR_ABOVE_CODE

    if np.any(inside):
        normalized = np.clip(
            (values[inside] - selected_minimum)
            / max(selected_maximum - selected_minimum, 1e-12),
            0.0,
            1.0,
        )
        bin_index = np.minimum(
            (normalized * (SCALAR_BINS - 1)).astype(np.uint16),
            SCALAR_BINS - 1,
        )
        codes[inside] = SCALAR_FIRST_COLOR_CODE + bin_index

    return codes


def create_scalar_context_lut(
    name,
    palette,
):
    """
    Discrete LUT matching encode_scalar_slice()'s codes: below/above
    range at indices 1/2 (index 0 is a dead slot never actually
    rendered, confirmed 2026-08), the property's own palette starting
    at index 3. Values outside the selected range are always fully
    transparent.
    """
    LUTLogic = _lut_logic_class()
    if LUTLogic is None:
        raise RuntimeError(
            "Dragonfly LUT creation API could not be located."
        )

    table_size = SCALAR_FIRST_COLOR_CODE + SCALAR_BINS

    try:
        lut = LUTLogic.createLUT(name)
    except Exception as exc:
        raise RuntimeError(
            f"Dragonfly could not create LUT {name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if lut is None:
        raise RuntimeError(
            f"Dragonfly returned None while creating LUT {name!r}."
        )

    try:
        lut.setIsDiscrete(True)
        lut.setTableSize(table_size)

        _set_color(lut, SCALAR_BACKGROUND_CODE, (0.0, 0.0, 0.0), 0.0)

        # Below/above range are always fully transparent.
        _set_color(
            lut,
            SCALAR_BELOW_CODE,
            (0.0, 0.0, 0.0),
            0.0,
        )
        # Duplicated onto two consecutive indices (2 AND 3), same fix
        # already applied to index 0 -- confirmed 2026-08 that index 2
        # alone was unreliable here too (getColorAtIndex(2) on the
        # "effective" LUT showed the wrong colour -- the start of the
        # gradient palette -- even though the source LUT was built
        # correctly). The gradient now starts one index later, at 4.
        _set_color(
            lut,
            SCALAR_ABOVE_CODE,
            (0.0, 0.0, 0.0),
            0.0,
        )
        _set_color(
            lut,
            SCALAR_ABOVE_CODE + 1,
            (0.0, 0.0, 0.0),
            0.0,
        )

        for bin_index in range(SCALAR_BINS):
            normalized = bin_index / float(SCALAR_BINS - 1)
            rgb = _interpolate_palette(normalized, palette)
            _set_color(
                lut, SCALAR_FIRST_COLOR_CODE + bin_index, rgb, 1.0
            )

    except Exception as exc:
        raise RuntimeError(
            f"LUT {name!r} was created, but its colors could not "
            f"be initialized: {type(exc).__name__}: {exc}"
        ) from exc

    return _finalize_lut(lut)


def create_geodect_lut(name="Geo_DECT 2D property map"):
    """
    Create the discrete two-property Geo_DECT LUT. Values outside the
    selected range are always fully transparent.
    """

    LUTLogic = _lut_logic_class()

    if LUTLogic is None:
        raise RuntimeError(
            "Dragonfly LUT creation API could not be located."
        )

    try:
        lut = LUTLogic.createLUT(name)
    except Exception as exc:
        raise RuntimeError(
            "Dragonfly could not create the Geo_DECT LUT: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if lut is None:
        raise RuntimeError(
            "Dragonfly's LUT creation method returned None."
        )

    try:
        lut.setIsDiscrete(True)
        lut.setTableSize(int(GEODECT_MAX_CODE + 1))

        _set_color(
            lut,
            GEODECT_BACKGROUND_CODE,
            (0.0, 0.0, 0.0),
            0.0
        )
        # Below/above range voxels are always fully transparent
        # (alpha=0.0), matching the background code -- not opaque
        # black/white. Confirmed bug (2026-08): alpha was hardcoded to
        # 1.0 here, so this rendered a solid opaque block instead of
        # true transparency.
        _set_color(
            lut,
            GEODECT_BELOW_CODE,
            (0.0, 0.0, 0.0),
            0.0,
        )
        _set_color(
            lut,
            GEODECT_ABOVE_CODE,
            (0.0, 0.0, 0.0),
            0.0,
        )

        # Use the 16-bit palette grid (subdivide_combined_palette_int16)
        # instead of interpolating with lab_to_rgb() 65025 times
        # separately. Dragonfly's Color API still needs 0-1 floats, so
        # the 16-bit grid is converted back only right before
        # setColorAtIndex() -- but the interpolation/quantization itself
        # goes through the 16-bit palette, not the old per-cell
        # lab_to_rgb interpolation.
        rgb_grid_16bit = subdivide_combined_palette_int16(GEODECT_BINS)
        rgb_grid = rgb_grid_16bit.astype(np.float64) / 65535.0

        for rhoe_bin in range(GEODECT_BINS):
            for zeff_bin in range(GEODECT_BINS):
                code = (
                    GEODECT_FIRST_COLOR_CODE
                    + rhoe_bin * GEODECT_BINS
                    + zeff_bin
                )
                _set_color(
                    lut,
                    code,
                    rgb_grid[rhoe_bin, zeff_bin],
                    1.0
                )

    except Exception as exc:
        raise RuntimeError(
            "The Geo_DECT LUT was created, but its colors could not "
            f"be initialized: {type(exc).__name__}: {exc}"
        ) from exc

    return _finalize_lut(lut)


def _interpolate_palette(normalized, palette):
    stops = np.asarray(palette, dtype=np.float64)
    position = np.clip(normalized, 0.0, 1.0) * (len(stops) - 1)
    lower = int(np.floor(position))
    upper = min(lower + 1, len(stops) - 1)
    fraction = position - lower
    return stops[lower] * (1.0 - fraction) + stops[upper] * fraction


def subdivide_palette_int16(palette, n_bins, input_max_value=255):
    """
    Subdivide a 1D RGB palette (a list of colour stops, such as
    ZEFF_PALETTE_RGB255 or RHOE_PALETTE_RGB255) into n_bins evenly
    spaced discrete colours, returned as 16-bit integers
    (uint16, 0-65535 per channel).

    Vectorized version (NumPy, single call) of the same logic as
    _interpolate_palette(), for any number of bins at once instead of
    one point at a time.

    :param palette: list/array of RGB colours, shape (N, 3), on the
                     input_max_value scale.
    :param n_bins: number of discrete colours to generate.
    :param input_max_value: scale of the input palette values
                             (255 for standard 8-bit RGB).
    :return: NumPy uint16 array, shape (n_bins, 3), values 0-65535.
    """
    stops = np.asarray(palette, dtype=np.float64)
    if stops.ndim != 2 or stops.shape[1] != 3:
        raise ValueError("palette must have shape (N, 3).")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")

    n_stops = stops.shape[0]
    positions = (
        np.array([0.5]) if n_bins == 1 else np.linspace(0.0, 1.0, n_bins)
    )

    scaled = positions * (n_stops - 1)
    lower_index = np.floor(scaled).astype(int)
    upper_index = np.minimum(lower_index + 1, n_stops - 1)
    fraction = (scaled - lower_index)[:, None]

    colors_float = (
        stops[lower_index] * (1.0 - fraction)
        + stops[upper_index] * fraction
    )

    scale = 65535.0 / float(input_max_value)
    colors_scaled = np.rint(colors_float * scale)
    colors_clipped = np.clip(colors_scaled, 0, 65535)
    return colors_clipped.astype(np.uint16)


def get_zeff_palette_16bit(n_bins=None):
    """Zeff palette subdivided into n_bins 16-bit colours (default: GEODECT_BINS, same resolution as "combined")."""
    return subdivide_palette_int16(
        ZEFF_PALETTE_RGB255, n_bins if n_bins is not None else GEODECT_BINS
    )


def get_rhoe_palette_16bit(n_bins=None):
    """rho_e palette subdivided into n_bins 16-bit colours (default: GEODECT_BINS, same resolution as "combined")."""
    return subdivide_palette_int16(
        RHOE_PALETTE_RGB255, n_bins if n_bins is not None else GEODECT_BINS
    )


def subdivide_combined_palette_int16(n_bins=None):
    """
    Generate the bivariate (Zeff x rho_e) colour grid used for
    "combined", as 16-bit integers (uint16, 0-65535 per channel) --
    vectorized NumPy version (a single call to lab_to_rgb for the whole
    grid), used by create_geodect_lut() instead of the former
    point-by-point interpolation.

    :param n_bins: number of subdivisions on EACH axis (Zeff and
                    rho_e). Default: GEODECT_BINS.
    :return: NumPy uint16 array, shape (n_bins, n_bins, 3) --
             colors[rhoe_bin, zeff_bin] = (R, G, B) in 0-65535.
             (same convention as create_geodect_lut:
             code = GEODECT_FIRST_COLOR_CODE + rhoe_bin * n_bins + zeff_bin)
    """
    if n_bins is None:
        n_bins = GEODECT_BINS
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")

    if n_bins == 1:
        zeff_n_axis = np.array([0.5])
        rhoe_n_axis = np.array([0.5])
    else:
        zeff_n_axis = np.linspace(0.0, 1.0, n_bins)
        rhoe_n_axis = np.linspace(0.0, 1.0, n_bins)

    zeff_grid, rhoe_grid = np.meshgrid(zeff_n_axis, rhoe_n_axis)
    a_grid = GEODECT_A_MIN + (GEODECT_A_MAX - GEODECT_A_MIN) * zeff_grid
    b_grid = GEODECT_B_MIN + (GEODECT_B_MAX - GEODECT_B_MIN) * rhoe_grid
    l_grid = np.full_like(a_grid, GEODECT_LIGHTNESS)

    rgb_float = lab_to_rgb(l_grid, a_grid, b_grid)  # (n_bins, n_bins, 3), 0-1

    rgb_scaled = np.rint(rgb_float * 65535.0)
    rgb_clipped = np.clip(rgb_scaled, 0, 65535)
    return rgb_clipped.astype(np.uint16)


def create_context_scalar_lut(
    name,
    data_minimum,
    data_maximum,
    selected_minimum,
    selected_maximum,
    palette,
    table_size=4096,
):
    """
    Create a continuous LUT preserving the full data context.

    Values outside the selected range are always fully transparent.
    The supplied palette is distributed through the selected range.
    """

    LUTLogic = _lut_logic_class()

    if LUTLogic is None:
        raise RuntimeError(
            "Dragonfly LUT creation API could not be located."
        )

    table_size = max(int(table_size), 2)

    data_minimum = float(data_minimum)
    data_maximum = float(data_maximum)
    selected_minimum = float(selected_minimum)
    selected_maximum = float(selected_maximum)

    if not np.isfinite(data_minimum):
        raise ValueError("data_minimum is not finite.")

    if not np.isfinite(data_maximum):
        raise ValueError("data_maximum is not finite.")

    if not np.isfinite(selected_minimum):
        raise ValueError("selected_minimum is not finite.")

    if not np.isfinite(selected_maximum):
        raise ValueError("selected_maximum is not finite.")

    if data_maximum <= data_minimum:
        data_maximum = data_minimum + 1.0e-12

    if selected_maximum <= selected_minimum:
        selected_maximum = selected_minimum + 1.0e-12

    try:
        lut = LUTLogic.createLUT(name)
    except Exception as exc:
        raise RuntimeError(
            f"Dragonfly could not create LUT {name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if lut is None:
        raise RuntimeError(
            f"Dragonfly returned None while creating LUT {name!r}."
        )

    try:
        # Keep the selected interval mapped across the full palette, exactly
        # as in the supplied working panel.  Use a dense discrete LUT so
        # Dragonfly samples the two endpoint RGBA entries exactly instead of
        # interpolating them into the palette.
        lut.setIsDiscrete(True)
        lut.setTableSize(table_size)

        data_denominator = max(
            data_maximum - data_minimum,
            1.0e-12
        )

        selected_denominator = max(
            selected_maximum - selected_minimum,
            1.0e-12
        )

        # IMPORTANT: this LUT is consumed in Dragonfly's native 2D
        # Window/Level mode.  Window/Level already maps the user-selected
        # physical interval onto the complete LUT domain.  Therefore the
        # palette must occupy the complete LUT interior, independently of the
        # absolute channel minimum/maximum.  The previous implementation put
        # the palette only at its physical-data positions and then applied a
        # second Window/Level remapping; for narrow ranges this collapsed Z_eff
        # into the first dark-blue part of the palette.
        #
        # Index 0 was confirmed (2026-08, live console test) to never
        # actually be rendered by Dragonfly's 2D WL colour mapping --
        # whatever colour it holds is ignored/skipped, the display uses
        # index 1's colour instead for anything that should have mapped
        # to index 0. This is why "below range" (originally at index 0)
        # always showed the background colour (effectively black)
        # instead of BELOW_RGB255, while "above range" (at the LAST
        # index, which IS rendered correctly) always worked fine.
        # Fix: duplicate the below-range colour onto BOTH index 0 (dead
        # slot, content irrelevant) and index 1 (the one actually used).
        # The palette interior now starts at index 2 instead of index 1.
        interior_denominator = max(table_size - 4, 1)
        for index in range(table_size):
            alpha = 1.0
            if index in (0, 1) or index == table_size - 1:
                rgb = np.asarray((0, 0, 0), dtype=np.float64)
                alpha = 0.0
            else:
                normalized = np.clip(
                    (index - 2) / float(interior_denominator),
                    0.0,
                    1.0,
                )
                rgb = _interpolate_palette(normalized, palette)

            _set_color(lut, index, rgb, alpha)

    except Exception as exc:
        raise RuntimeError(
            f"LUT {name!r} was created, but its colors could not "
            f"be initialized: {type(exc).__name__}: {exc}"
        ) from exc

    return _finalize_lut(lut)


def apply_lut_to_channel_in_layout(
    channel,
    lut,
    scene_layout,
    visual_channel=None,
    view=None,
    diagnostic=True,
):
    """Apply and verify a LUT in the active Dragonfly layout.

    Dragonfly 2025.1 accepts the genealogical LUT setters on the native
    Channel. The renderer, however, consumes properties associated with the
    VisualChannel in the current SceneLayout. This implementation therefore:

    1. applies the documented genealogical setters to the Channel;
    2. attempts any direct visual/layout setters exposed by this build;
    3. reads the effective 2D and 3D LUTs back through all available getters;
    4. records visual window/level and color-related state;
    5. keeps the ScalarValuesCollection diagnostic as contextual evidence,
       without creating or modifying scalar slots.
    """

    channel_key = None
    try:
        channel_key = channel.getGUID()
    except Exception:
        channel_key = id(channel)

    now = time.monotonic()
    last_call_time = _LUT_APPLICATION_STATE["last_call_time_by_channel"].get(
        channel_key
    )
    if (
        last_call_time is not None
        and (now - last_call_time) < _LUT_APPLICATION_MIN_INTERVAL_SECONDS
    ):
        return "SKIPPED_THROTTLED"

    if _LUT_APPLICATION_STATE["in_progress"]:
        return "SKIPPED_REENTRANT"

    _LUT_APPLICATION_STATE["in_progress"] = True
    _LUT_APPLICATION_STATE["last_call_time_by_channel"][channel_key] = now
    try:
        return _apply_lut_to_channel_in_layout_impl(
            channel=channel,
            lut=lut,
            scene_layout=scene_layout,
            visual_channel=visual_channel,
            view=view,
            diagnostic=diagnostic,
        )
    finally:
        _LUT_APPLICATION_STATE["in_progress"] = False


def _apply_lut_to_channel_in_layout_impl(
    channel,
    lut,
    scene_layout,
    visual_channel=None,
    view=None,
    diagnostic=True,
):
    """Actual implementation, called only through the guard above."""

    diagnostic_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "geodect_lut_diagnostic.txt",
    )
    # Truncated at the start of every call (instead of appending forever)
    # so this file always holds just the most recent report, not an
    # ever-growing history across a whole working session.
    _diagnostic_write_state = {"truncated": False}

    def diagnostic_write(message):
        try:
            mode = "a" if _diagnostic_write_state["truncated"] else "w"
            with open(diagnostic_path, mode, encoding="utf-8") as handle:
                handle.write(str(message))
                handle.write("\n")
            _diagnostic_write_state["truncated"] = True
        except Exception:
            pass

    def filtered_names(obj, keywords=None):
        if obj is None:
            return []
        if keywords is None:
            keywords = (
                "lut", "lookup", "color", "colour", "palette",
                "window", "level", "range", "scalar", "value",
                "layout", "visual", "display", "render", "property",
                "uuid", "genealog", "mode", "tag",
            )
        try:
            return sorted(
                name for name in dir(obj)
                if not name.startswith("__")
                and any(key in name.lower() for key in keywords)
            )
        except Exception:
            return ["<dir unavailable>"]

    def object_uuid(obj):
        if obj is None:
            return None
        logic = _lut_logic_class()
        if logic is not None:
            try:
                value = logic.getLUTUUID(obj)
                if value:
                    return str(value)
            except Exception:
                pass
        for method_name in (
            "getLUTUUID", "getLutUUID", "getUUID", "getGUID", "getUid",
        ):
            method = getattr(obj, method_name, None)
            if callable(method):
                try:
                    value = method()
                    if value:
                        return str(value)
                except Exception:
                    pass
        return None

    def call_attempt(label, callable_object, variants):
        """Try positional/keyword variants and log every result."""
        if not callable(callable_object):
            diagnostic_write(label + ": unavailable")
            return None, False

        last_error = None
        for index, variant in enumerate(variants, 1):
            args = variant.get("args", ())
            kwargs = variant.get("kwargs", {})
            try:
                result = callable_object(*args, **kwargs)
                diagnostic_write(
                    "{} variant {}: SUCCESS; result={!r}; type={!r}".format(
                        label, index, result, type(result)
                    )
                )
                return result, True
            except Exception as exc:
                last_error = exc
                diagnostic_write(
                    "{} variant {}: {}: {}".format(
                        label, index, type(exc).__name__, exc
                    )
                )
        if last_error is not None:
            diagnostic_write(
                "{}: all variants failed\n{}".format(
                    label, traceback.format_exc()
                )
            )
        return None, False

    def inspect_scalar_context(label, owner):
        """Read scalar state only; do not create or modify collections."""
        diagnostic_write("")
        diagnostic_write("=== {} ===".format(label))
        if owner is None:
            diagnostic_write("Owner is None")
            return
        getter = getattr(owner, "getScalarValuesCollection", None)
        if not callable(getter):
            diagnostic_write("getScalarValuesCollection unavailable")
            return
        try:
            collection = getter()
            diagnostic_write("Collection: {!r}".format(collection))
            diagnostic_write("Collection type: {!r}".format(type(collection)))
        except Exception:
            diagnostic_write(
                "getScalarValuesCollection failed:\n{}".format(
                    traceback.format_exc()
                )
            )
            return
        if collection is None:
            diagnostic_write("No scalar collection attached.")
            return
        for method_name in (
            "getSlotCount", "getScalarValuesSize", "getUseScalarValues",
            "getTSize", "getDataTypes",
        ):
            method = getattr(collection, method_name, None)
            if not callable(method):
                diagnostic_write(method_name + ": unavailable")
                continue
            try:
                value = method()
                diagnostic_write(
                    "{}: {!r}; type={!r}".format(
                        method_name, value, type(value)
                    )
                )
                if method_name == "getDataTypes" and value is not None:
                    size_method = getattr(value, "getSize", None)
                    if callable(size_method):
                        try:
                            diagnostic_write(
                                "getDataTypes().getSize(): {!r}".format(
                                    size_method()
                                )
                            )
                        except Exception:
                            pass
            except Exception:
                diagnostic_write(
                    "{} failed:\n{}".format(
                        method_name, traceback.format_exc()
                    )
                )

    def inspect_visual_state(stage):
        diagnostic_write("")
        diagnostic_write("=== VISUAL STATE: {} ===".format(stage))
        if visual_channel is None:
            diagnostic_write("VisualChannel is None")
            return
        diagnostic_write(
            "Visual relevant members: {}".format(
                filtered_names(visual_channel)
            )
        )
        no_arg_methods = (
            "get2DWindowLevelValuesForAllViews",
            "get2DWindowLevel2ValuesForAllViews",
            "get2DWindowLevelValuesNormalizedForAllViews",
            "get2DWindowLevel2ValuesNormalizedForAllViews",
            "get3DWindowLevelValuesForAllViews",
            "get3DWindowLevelValuesNormalizedForAllViews",
            "getIsoValueForAllViews",
        )
        for method_name in no_arg_methods:
            method = getattr(visual_channel, method_name, None)
            if not callable(method):
                continue
            try:
                value = method()
                diagnostic_write("{}: {!r}".format(method_name, value))
            except Exception:
                diagnostic_write(
                    "{} failed: {}".format(
                        method_name, traceback.format_exc()
                    )
                )

    api = _view_api()
    if api is None:
        diagnostic_write("ERROR: _view_api() returned None")
        return "API"
    if channel is None:
        diagnostic_write("ERROR: channel is None")
        return "CHANNEL"
    if lut is None:
        diagnostic_write("ERROR: LUT is None")
        return "LUT IS NONE"
    if scene_layout is None:
        diagnostic_write("ERROR: scene_layout is None")
        return "LAYOUT"

    LayoutPropertiesHelper = api["LayoutPropertiesHelper"]

    layout_full_name = None
    for method_name in ("getFullName", "getGenealogicalName", "getFullNameWithPath"):
        method = getattr(scene_layout, method_name, None)
        if not callable(method):
            continue
        try:
            value = method()
            if value:
                layout_full_name = str(value)
                break
        except Exception:
            diagnostic_write(
                "{} failed:\n{}".format(method_name, traceback.format_exc())
            )
    if not layout_full_name:
        diagnostic_write("ERROR: unable to resolve layout name")
        return "LAYOUT NAME ERROR"

    lut_uuid = object_uuid(lut)
    if not lut_uuid:
        diagnostic_write("ERROR: LUT UUID not found")
        diagnostic_write("LUT members: {}".format(filtered_names(lut)))
        return "LUT UUID ERROR"

    # Fast runtime route used by the live range controls. The full diagnostic
    # route below intentionally tries many API variants and read-backs; that is
    # useful for troubleshooting but far too expensive for interactive use.
    if not diagnostic:
        try:
            LayoutPropertiesHelper.set3DLUTUUIDFromGenealogicalName(
                layoutFullName=layout_full_name,
                anObject=channel,
                lutUUID=lut_uuid,
                aScalarValueTypeTag="",
            )
            LayoutPropertiesHelper.set2DLUTUUIDFromGenealogicalName(
                layoutFullName=layout_full_name,
                anObject=channel,
                lutUUID=lut_uuid,
                aScalarValueTypeTag="",
            )
        except Exception as exc:
            return "LUT ASSIGNMENT ERROR: {}: {}".format(
                type(exc).__name__, exc
            )

        if visual_channel is not None:
            try:
                visual_channel.setPropertyDirty()
            except Exception:
                pass
            try:
                visual_channel.refreshAllParentViewsIfVisible()
            except Exception:
                pass
        if view is not None:
            try:
                view.refresh()
            except Exception:
                pass
        return "OK"

    diagnostic_write("")
    diagnostic_write("######## GEO_DECT LUT DIAGNOSTIC ########")
    diagnostic_write("Loaded module: {}".format(__file__))
    diagnostic_write("Layout full name: {!r}".format(layout_full_name))
    diagnostic_write("Scene layout: {!r}".format(scene_layout))
    diagnostic_write("Scene layout type: {!r}".format(type(scene_layout)))
    diagnostic_write(
        "Scene layout relevant members: {}".format(filtered_names(scene_layout))
    )
    diagnostic_write("Requested LUT: {!r}".format(lut))
    diagnostic_write("Requested LUT UUID: {!r}".format(lut_uuid))
    diagnostic_write("Channel: {!r}".format(channel))
    diagnostic_write("Channel type: {!r}".format(type(channel)))
    diagnostic_write("VisualChannel: {!r}".format(visual_channel))
    diagnostic_write("VisualChannel type: {!r}".format(type(visual_channel)))
    diagnostic_write(
        "LayoutPropertiesHelper relevant members: {}".format(
            filtered_names(LayoutPropertiesHelper)
        )
    )

    inspect_scalar_context("CHANNEL SCALAR CONTEXT", channel)
    inspect_scalar_context("VISUAL SCALAR CONTEXT", visual_channel)
    inspect_visual_state("BEFORE LUT ASSIGNMENT")

    assignment_success = False

    # Primary, known-compatible route for Dragonfly 2025.1.
    try:
        LayoutPropertiesHelper.set3DLUTUUIDFromGenealogicalName(
            layoutFullName=layout_full_name,
            anObject=channel,
            lutUUID=lut_uuid,
            aScalarValueTypeTag="",
        )
        diagnostic_write("set3DLUTUUIDFromGenealogicalName: SUCCESS")
        assignment_success = True
    except Exception:
        diagnostic_write(
            "set3DLUTUUIDFromGenealogicalName failed:\n{}".format(
                traceback.format_exc()
            )
        )

    try:
        LayoutPropertiesHelper.set2DLUTUUIDFromGenealogicalName(
            layoutFullName=layout_full_name,
            anObject=channel,
            lutUUID=lut_uuid,
            aScalarValueTypeTag="",
        )
        diagnostic_write("set2DLUTUUIDFromGenealogicalName: SUCCESS")
        assignment_success = True
    except Exception:
        diagnostic_write(
            "set2DLUTUUIDFromGenealogicalName failed:\n{}".format(
                traceback.format_exc()
            )
        )

    # Prefer passing the actual LUT object. In Dragonfly 2025.1 the helper
    # signatures observed at runtime use (layout, object, value, tag), and the
    # object that owns the genealogical properties is the native Channel.
    for dimension in ("3D", "2D"):
        object_setter = getattr(
            LayoutPropertiesHelper,
            "set{}LUTAsLUT".format(dimension),
            None,
        )
        _result, ok = call_attempt(
            "LayoutPropertiesHelper.set{}LUTAsLUT".format(dimension),
            object_setter,
            (
                {"args": (scene_layout, channel, lut, "")},
                {"args": (scene_layout, channel, lut)},
                {"args": (scene_layout, visual_channel, lut, "")},
                {"kwargs": {
                    "layout": scene_layout,
                    "anObject": channel,
                    "lut": lut,
                    "aScalarValueTypeTag": "",
                }},
            ),
        )
        assignment_success = assignment_success or ok

        genealogical_object_setter = getattr(
            LayoutPropertiesHelper,
            "set{}LUTAsLUTFromGenealogicalName".format(dimension),
            None,
        )
        _result, ok = call_attempt(
            "LayoutPropertiesHelper.set{}LUTAsLUTFromGenealogicalName".format(
                dimension
            ),
            genealogical_object_setter,
            (
                {"args": (layout_full_name, channel, lut, "")},
                {"args": (layout_full_name, channel, lut)},
                {"kwargs": {
                    "layoutFullName": layout_full_name,
                    "anObject": channel,
                    "lut": lut,
                    "aScalarValueTypeTag": "",
                }},
            ),
        )
        assignment_success = assignment_success or ok

        uuid_setter = getattr(
            LayoutPropertiesHelper,
            "set{}LUTUUID".format(dimension),
            None,
        )
        _result, ok = call_attempt(
            "LayoutPropertiesHelper.set{}LUTUUID".format(dimension),
            uuid_setter,
            (
                {"args": (scene_layout, channel, lut_uuid, "")},
                {"args": (scene_layout, channel, lut_uuid)},
                {"args": (scene_layout, visual_channel, lut_uuid, "")},
                {"kwargs": {
                    "layout": scene_layout,
                    "anObject": channel,
                    "lutUUID": lut_uuid,
                    "aScalarValueTypeTag": "",
                }},
            ),
        )
        assignment_success = assignment_success or ok

    # Read back through the same native Channel route used by the setters.
    # A non-None object whose UUID matches the requested LUT is the decisive
    # evidence that the layout property was actually stored.
    readback_values = {}
    readback_match = False

    for dimension in ("3D", "2D"):
        getter_name = "get{}LUT".format(dimension)
        result, ok = call_attempt(
            "LayoutPropertiesHelper.{}".format(getter_name),
            getattr(LayoutPropertiesHelper, getter_name, None),
            (
                {"args": (scene_layout, channel, "")},
                {"args": (scene_layout, channel)},
                {"args": (scene_layout, visual_channel, "")},
                {"kwargs": {
                    "layout": scene_layout,
                    "anObject": channel,
                    "aScalarValueTypeTag": "",
                }},
            ),
        )
        if ok:
            result_uuid = object_uuid(result)
            readback_values[getter_name] = result
            readback_values[getter_name + "_uuid"] = result_uuid
            diagnostic_write(
                "{} read-back UUID: {!r}; requested={!r}; match={!r}".format(
                    getter_name,
                    result_uuid,
                    lut_uuid,
                    result_uuid == lut_uuid,
                )
            )
            readback_match = readback_match or (result_uuid == lut_uuid)

        genealogical_getter_name = "{}FromGenealogicalName".format(
            getter_name
        )
        result, ok = call_attempt(
            "LayoutPropertiesHelper.{}".format(genealogical_getter_name),
            getattr(
                LayoutPropertiesHelper,
                genealogical_getter_name,
                None,
            ),
            (
                {"args": (layout_full_name, channel, "")},
                {"args": (layout_full_name, channel)},
                {"kwargs": {
                    "layoutFullName": layout_full_name,
                    "anObject": channel,
                    "aScalarValueTypeTag": "",
                }},
            ),
        )
        if ok:
            result_uuid = object_uuid(result)
            readback_values[genealogical_getter_name] = result
            readback_values[genealogical_getter_name + "_uuid"] = result_uuid
            diagnostic_write(
                "{} read-back UUID: {!r}; requested={!r}; match={!r}".format(
                    genealogical_getter_name,
                    result_uuid,
                    lut_uuid,
                    result_uuid == lut_uuid,
                )
            )
            readback_match = readback_match or (result_uuid == lut_uuid)

        source_getter_name = "get{}LUTSource".format(dimension)
        result, ok = call_attempt(
            "LayoutPropertiesHelper.{}".format(source_getter_name),
            getattr(LayoutPropertiesHelper, source_getter_name, None),
            (
                {"args": (scene_layout, channel, "")},
                {"args": (scene_layout, channel)},
                {"args": (scene_layout, visual_channel, "")},
                {"kwargs": {
                    "layout": scene_layout,
                    "anObject": channel,
                    "aScalarValueTypeTag": "",
                }},
            ),
        )
        if ok:
            readback_values[source_getter_name] = result

        mode_getter_name = "getColorRangeMode{}".format(dimension)
        result, ok = call_attempt(
            "LayoutPropertiesHelper.{}".format(mode_getter_name),
            getattr(LayoutPropertiesHelper, mode_getter_name, None),
            (
                {"args": (scene_layout, channel, "")},
                {"args": (scene_layout, channel)},
                {"args": (scene_layout, visual_channel, "")},
            ),
        )
        if ok:
            readback_values[mode_getter_name] = result

    # Ensure 2D rendering consumes LUT alpha. Without this, transparent
    # endpoints can appear as the Zeff palette endpoint (cyan/green).
    try:
        layout_name = scene_layout.getGenealogicalName()
        LayoutPropertiesHelper.setUse2DLUTAlphaFromGenealogicalName(
            layout_name, channel, True
        )
        try:
            LayoutPropertiesHelper.setUse2DLUTAlphaFromGenealogicalName(
                layout_name, visual_channel, True
            )
        except Exception:
            pass
        diagnostic_write(
            "setUse2DLUTAlphaFromGenealogicalName: SUCCESS "
            "(Channel + VisualChannel attempt)"
        )
    except Exception as exc:
        diagnostic_write(
            "setUse2DLUTAlphaFromGenealogicalName: {}: {}".format(
                type(exc).__name__, exc
            )
        )

    # Dragonfly 2025.1 may clone the supplied LUT into private per-layout
    # objects. Copy colors and alpha into the effective clones too.
    #
    # Confirmed in the Dragonfly Python console (2026-08) that
    # LookupTable.copyColorSettingsFrom(source) copies the full table
    # (including table size and alpha) in a single native call, without
    # touching the target's own object identity (title/GUID) the way
    # copyFrom() would. This replaces the previous manual per-index
    # Python loop, which was the suspected source of colors/alpha being
    # dropped on the internal Dragonfly clone.
    def copy_lut_colors(target):
        if target is None or target is lut:
            return False

        native_copy = getattr(target, "copyColorSettingsFrom", None)
        if callable(native_copy):
            try:
                native_copy(lut)
                for method_name in ("setDataDirty", "setPropertyDirty"):
                    method = getattr(target, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception:
                            pass
                return True
            except Exception as exc:
                diagnostic_write(
                    "copyColorSettingsFrom failed, falling back to "
                    "manual per-index copy: {}: {}".format(
                        type(exc).__name__, exc
                    )
                )

        # Fallback for Dragonfly versions where copyColorSettingsFrom is
        # not available on the target object.
        get_size = getattr(lut, "getTableSize", None)
        get_color = getattr(lut, "getColorAtIndex", None)
        set_size = getattr(target, "setTableSize", None)
        set_color = getattr(target, "setColorAtIndex", None)
        if not all(callable(x) for x in (get_size, get_color, set_color)):
            return False
        size = int(get_size())
        if callable(set_size):
            try:
                set_size(size)
            except Exception:
                pass

        source_is_discrete = getattr(lut, "getIsDiscrete", None)
        target_set_discrete = getattr(target, "setIsDiscrete", None)
        if callable(source_is_discrete) and callable(target_set_discrete):
            try:
                target_set_discrete(bool(source_is_discrete()))
            except Exception:
                pass

        for i in range(size):
            set_color(i, get_color(i))
        for method_name in ("setDataDirty", "setPropertyDirty"):
            method = getattr(target, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
        return True

    for label, getter in (
        ("effective2D", getattr(LayoutPropertiesHelper, "get2DLUT", None)),
        ("effective3D", getattr(LayoutPropertiesHelper, "get3DLUT", None)),
    ):
        if callable(getter):
            try:
                effective = getter(scene_layout, visual_channel)
                copied = copy_lut_colors(effective)
                diagnostic_write(
                    "{} LUT color copy: {}".format(
                        label, "SUCCESS" if copied else "SKIPPED"
                    )
                )
            except Exception as exc:
                diagnostic_write(
                    "{} LUT color copy: {}: {}".format(
                        label, type(exc).__name__, exc
                    )
                )


    # NOTE (2026-08): forcing ColorRangeMode3D to WL (to match 2D) and/or
    # OpacityMode3D to match 2D repeatedly froze Dragonfly's 3D renderer on
    # this installation, reproducibly, even on a small test volume. This is
    # left deliberately untouched for now -- the 3D volume stays in its
    # default color range mode (grey) until this is understood to be safe.
    # The 2D views are correctly colored without this synchronization.

    diagnostic_write("Read-back summary: {!r}".format(readback_values))
    diagnostic_write("Any requested LUT read-back match: {!r}".format(
        readback_match
    ))

    # Ask Dragonfly to propagate the changed layout properties before the
    # visual refresh. These calls are optional and build-dependent.
    for label, owner, method_name in (
        ("scene_layout.setPropertyDirty", scene_layout, "setPropertyDirty"),
        (
            "scene_layout.propagateLayoutUpdated",
            scene_layout,
            "propagateLayoutUpdated",
        ),
    ):
        method = getattr(owner, method_name, None)
        if callable(method):
            try:
                method()
                diagnostic_write(label + ": SUCCESS")
            except Exception:
                diagnostic_write(
                    label + " failed:\n" + traceback.format_exc()
                )

    # Preserve the physical window/level installed by the caller.
    # Resetting to the native full range here destroyed the preview-matched
    # mapping for Z_eff and rho_e and made the indexed combined volume vanish.
    if visual_channel is not None:
        diagnostic_write(
            "Preserving caller-supplied physical window/level: SUCCESS"
        )

        try:
            visual_channel.setPropertyDirty()
            diagnostic_write("visual_channel.setPropertyDirty: SUCCESS")
        except Exception:
            diagnostic_write(
                "setPropertyDirty failed:\n{}".format(traceback.format_exc())
            )

        for method_name in (
            "refreshAllParentViewsIfVisible",
            "refreshAllParentViews",
        ):
            method = getattr(visual_channel, method_name, None)
            if callable(method):
                try:
                    method()
                    diagnostic_write(method_name + ": SUCCESS")
                except Exception:
                    diagnostic_write(
                        method_name + " failed:\n" + traceback.format_exc()
                    )

    if view is not None:
        try:
            view.refresh()
            diagnostic_write("view.refresh: SUCCESS")
        except Exception:
            diagnostic_write(
                "view.refresh failed:\n{}".format(traceback.format_exc())
            )

    inspect_visual_state("AFTER LUT ASSIGNMENT")
    diagnostic_write("######## END GEO_DECT LUT DIAGNOSTIC ########")

    if not assignment_success:
        return "LUT ASSIGNMENT ERROR"
    return "OK"

def diagnose_current_view(channel):
    """Inspect the current Dragonfly view and scene layout."""

    print("\n" + "=" * 70)
    print("GEODECT VIEW DIAGNOSTIC")
    print("=" * 70)
    print("Channel:", channel)
    print("Channel title:", channel.getTitle())

    api = _view_api()
    if api is None:
        print("ERROR: view API could not be imported.")
        return

    WorkingContext = api["WorkingContext"]
    LayoutHelper = api["LayoutHelper"]
    VisualChannel = api["VisualChannel"]

    try:
        view = WorkingContext.getCurrentView(None)
    except Exception as exc:
        print("getCurrentView failed:", repr(exc))
        return

    print("\nCurrent view:")
    print("  type:", type(view))
    print("  repr:", repr(view))

    if view is None:
        print("No current view.")
        return

    print("\nView methods containing useful keywords:")
    for name in sorted(dir(view)):
        lower = name.lower()
        if any(word in lower for word in (
            "visual", "scene", "layout", "channel",
            "object", "node", "renderer"
        )):
            print(" ", name)

    try:
        scene_layout = LayoutHelper.getSceneLayoutOfView(view)
    except Exception as exc:
        print("\ngetSceneLayoutOfView failed:", repr(exc))
        return

    print("\nScene layout:")
    print("  type:", type(scene_layout))
    print("  repr:", repr(scene_layout))

    print("\nScene-layout methods containing useful keywords:")
    for name in sorted(dir(scene_layout)):
        lower = name.lower()
        if any(word in lower for word in (
            "visual", "scene", "layout", "channel",
            "object", "node", "renderer", "child"
        )):
            print(" ", name)

    print("\nVisualChannel class name:")
    try:
        print(" ", VisualChannel.getClassNameStatic())
    except Exception as exc:
        print("  failed:", repr(exc))

    print("\nTrying likely scene-layout getters:")

    candidate_methods = (
        "getChildren",
        "getChildrenNodes",
        "getAllChildren",
        "getAllNodes",
        "getNodes",
        "getObjects",
        "getVisuals",
        "getVisualChannels",
        "getChannels",
    )

    for method_name in candidate_methods:
        method = getattr(scene_layout, method_name, None)
        if not callable(method):
            continue

        try:
            result = method()
            print(f"\n{method_name}():")
            print("  type:", type(result))
            print("  repr:", repr(result))

            try:
                for index, item in enumerate(result):
                    print(
                        f"  [{index}] type={type(item)} "
                        f"repr={repr(item)}"
                    )
            except TypeError:
                pass

        except Exception as exc:
            print(f"\n{method_name}() failed:", repr(exc))

    print("\nDiagnostic complete.")
    print("=" * 70)

def finish_channel(channel, minimum=None, maximum=None):
    """Validate, configure and publish a Channel created by fromNDArray."""
    if channel is None:
        raise ValueError("Cannot publish a null Dragonfly channel.")
    if not channel.getIsDataInitialized():
        raise RuntimeError(
            f"Channel {channel.getTitle()!r} is not initialized."
        )

    native_min = float(channel.getMinimumValue())
    native_max = float(channel.getMaximumValue())
    if not np.isfinite(native_min) or not np.isfinite(native_max):
        raise RuntimeError(
            f"Channel {channel.getTitle()!r} has non-finite native limits: "
            f"{native_min}, {native_max}."
        )

    if minimum is not None and maximum is not None:
        minimum = float(minimum)
        maximum = float(maximum)
        if np.isfinite(minimum) and np.isfinite(maximum) and maximum > minimum:
            width = maximum - minimum
            center = (maximum + minimum) / 2.0
            channel.addSuggestedWindowLevelValues(width, center)
            try:
                channel.setInitialWindowLevelValues(width, center)
            except Exception:
                pass

    channel.setDataDirty()
    channel.publish()
    return channel