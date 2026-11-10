import numpy as np


# Single source of truth for preview and Dragonfly LUT generation.
ZEFF_PALETTE_RGB255 = (
    (5, 20, 80),
    (0, 110, 190),
    (0, 210, 220),
    (170, 245, 170),
)

RHOE_PALETTE_RGB255 = (
    (90, 0, 0),
    (190, 35, 15),
    (245, 125, 20),
    (255, 235, 80),
)

GEODECT_LIGHTNESS = 68.0
GEODECT_A_MIN = -55.0
GEODECT_A_MAX = 55.0
GEODECT_B_MIN = -55.0
GEODECT_B_MAX = 55.0

BACKGROUND_RGB255 = (0, 0, 0)
BELOW_RGB255 = (100, 100, 100)
ABOVE_RGB255 = (255, 255, 255)


def lab_to_rgb(lightness, a_channel, b_channel):
    """Convert CIE L*a*b* arrays to clipped sRGB values in [0, 1]."""
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
            3.0 * delta ** 2 * (component - 4.0 / 29.0),
        )

    x = 0.95047 * inverse_f(fx)
    y = 1.00000 * inverse_f(fy)
    z = 1.08883 * inverse_f(fz)

    linear = np.stack(
        (
            3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
            -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
            0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
        ),
        axis=-1,
    )
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def normalize_property(values, value_min, value_max):
    """
    Normalize an array between 0 and 1.

    Values below value_min are clipped to 0.
    Values above value_max are clipped to 1.
    NaN values remain NaN.
    """

    values = np.asarray(
        values,
        dtype=np.float32
    )

    value_min = float(value_min)
    value_max = float(value_max)

    if value_max <= value_min:
        raise ValueError(
            "The maximum value must be greater than "
            "the minimum value."
        )

    normalized = (
        values - value_min
    ) / (
        value_max - value_min
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    normalized[~np.isfinite(values)] = np.nan

    return normalized.astype(
        np.float32,
        copy=False
    )

def classify_display_range(
    values,
    value_min,
    value_max
):
    """
    Classify values relative to a selected display range.

    Returns
    -------
    dict
        Boolean masks:

        valid:
            finite values

        below:
            finite values below value_min

        inside:
            finite values between value_min and value_max,
            including the limits

        above:
            finite values above value_max

        invalid:
            NaN or infinite values
    """

    values = np.asarray(
        values,
        dtype=np.float32
    )

    value_min = float(value_min)
    value_max = float(value_max)

    if value_max <= value_min:
        raise ValueError(
            "The maximum value must be greater than "
            "the minimum value."
        )

    valid = np.isfinite(values)

    below = (
        valid
        & (values < value_min)
    )

    inside = (
        valid
        & (values >= value_min)
        & (values <= value_max)
    )

    above = (
        valid
        & (values > value_max)
    )

    invalid = ~valid

    return {
        "valid": valid,
        "below": below,
        "inside": inside,
        "above": above,
        "invalid": invalid,
    }

def interpolate_rgb(
    normalized_values,
    start_rgb,
    middle_rgb,
    end_rgb
):
    """
    Create a three-colour sequential RGB map.

    normalized_values must contain values between 0 and 1.
    """

    normalized_values = np.asarray(
        normalized_values,
        dtype=np.float32
    )

    start_rgb = np.asarray(
        start_rgb,
        dtype=np.float32
    )

    middle_rgb = np.asarray(
        middle_rgb,
        dtype=np.float32
    )

    end_rgb = np.asarray(
        end_rgb,
        dtype=np.float32
    )

    rgb = np.zeros(
        normalized_values.shape + (3,),
        dtype=np.float32
    )

    lower_half = (
        np.isfinite(normalized_values)
        & (normalized_values <= 0.5)
    )

    upper_half = (
        np.isfinite(normalized_values)
        & (normalized_values > 0.5)
    )

    lower_position = (
        normalized_values[lower_half] / 0.5
    )

    rgb[lower_half] = (
        start_rgb
        + lower_position[:, None]
        * (middle_rgb - start_rgb)
    )

    upper_position = (
        (normalized_values[upper_half] - 0.5)
        / 0.5
    )

    rgb[upper_half] = (
        middle_rgb
        + upper_position[:, None]
        * (end_rgb - middle_rgb)
    )

    invalid = ~np.isfinite(normalized_values)

    rgb[invalid] = 0.0

    return np.clip(
        rgb,
        0.0,
        255.0
    ).astype(
        np.uint8
    )




def interpolate_palette(normalized_values, palette):
    """Interpolate an arbitrary RGB palette for normalized values in [0, 1]."""
    normalized_values = np.asarray(normalized_values, dtype=np.float32)
    stops = np.asarray(palette, dtype=np.float32)
    if stops.ndim != 2 or stops.shape[1] != 3 or len(stops) < 2:
        raise ValueError("palette must contain at least two RGB triples.")

    finite = np.isfinite(normalized_values)
    safe = np.where(finite, np.clip(normalized_values, 0.0, 1.0), 0.0)
    position = safe * (len(stops) - 1)
    lower = np.floor(position).astype(np.int32)
    lower = np.clip(lower, 0, len(stops) - 1)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    fraction = (position - lower)[..., None]
    rgb = stops[lower] * (1.0 - fraction) + stops[upper] * fraction
    rgb[~finite] = 0.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def create_zeff_rgb(
    zeff,
    zeff_min,
    zeff_max,
    background_rgb=(0, 0, 0)
):
    """
    Create a cold sequential colour image for Zeff.

    Only values inside the selected range are coloured; everything
    else (below range, above range, invalid) becomes background_rgb
    (transparent when combined with an alpha channel downstream).
    """

    zeff = np.asarray(
        zeff,
        dtype=np.float32
    )

    masks = classify_display_range(
        zeff,
        zeff_min,
        zeff_max
    )

    normalized = normalize_property(
        zeff,
        zeff_min,
        zeff_max
    )

    colour_image = interpolate_palette(
        normalized,
        ZEFF_PALETTE_RGB255,
    )

    rgb = np.empty(
        zeff.shape + (3,),
        dtype=np.uint8
    )

    rgb[:] = np.asarray(
        background_rgb,
        dtype=np.uint8
    )

    rgb[masks["inside"]] = (
        colour_image[masks["inside"]]
    )

    rgb[masks["invalid"]] = np.asarray(
        background_rgb,
        dtype=np.uint8
    )

    return rgb


def create_rhoe_rgb(
    rhoe,
    rhoe_min,
    rhoe_max,
    background_rgb=(0, 0, 0)
):
    """
    Create a warm sequential colour image for rho_e.

    Only values inside the selected range are coloured; everything
    else (below range, above range, invalid) becomes background_rgb
    (transparent when combined with an alpha channel downstream).
    """

    rhoe = np.asarray(
        rhoe,
        dtype=np.float32
    )

    masks = classify_display_range(
        rhoe,
        rhoe_min,
        rhoe_max
    )

    normalized = normalize_property(
        rhoe,
        rhoe_min,
        rhoe_max
    )

    colour_image = interpolate_palette(
        normalized,
        RHOE_PALETTE_RGB255,
    )

    rgb = np.empty(
        rhoe.shape + (3,),
        dtype=np.uint8
    )

    rgb[:] = np.asarray(
        background_rgb,
        dtype=np.uint8
    )

    rgb[masks["inside"]] = (
        colour_image[masks["inside"]]
    )

    rgb[masks["invalid"]] = np.asarray(
        background_rgb,
        dtype=np.uint8
    )

    return rgb


def create_geodect_rgb(
    zeff,
    rhoe,
    zeff_min,
    zeff_max,
    rhoe_min,
    rhoe_max,
    background_rgb=(0, 0, 0)
):
    """
    Create the combined Geo_DECT image using CIE L*a*b*.

    A voxel is considered inside the selected range only if
    both Zeff and rho_e are inside their respective ranges.
    Only such voxels are shown; everything else (below/above
    either range, invalid) becomes background_rgb (transparent
    when combined with an alpha channel downstream).
    """

    zeff = np.asarray(
        zeff,
        dtype=np.float32
    )

    rhoe = np.asarray(
        rhoe,
        dtype=np.float32
    )

    if zeff.shape != rhoe.shape:
        raise ValueError(
            "zeff and rhoe must have the same shape."
        )

    zeff_masks = classify_display_range(
        zeff,
        zeff_min,
        zeff_max
    )

    rhoe_masks = classify_display_range(
        rhoe,
        rhoe_min,
        rhoe_max
    )

    valid = (
        zeff_masks["valid"]
        & rhoe_masks["valid"]
    )

    inside = (
        zeff_masks["inside"]
        & rhoe_masks["inside"]
    )

    invalid = ~valid

    zeff_normalized = normalize_property(
        zeff,
        zeff_min,
        zeff_max
    )

    rhoe_normalized = normalize_property(
        rhoe,
        rhoe_min,
        rhoe_max
    )

    lab_image = np.zeros(
        zeff.shape + (3,),
        dtype=np.float32
    )

    lab_image[..., 0] = GEODECT_LIGHTNESS

    lab_image[..., 1] = (
        GEODECT_A_MIN
        + zeff_normalized * (GEODECT_A_MAX - GEODECT_A_MIN)
    )

    lab_image[..., 2] = (
        GEODECT_B_MIN
        + rhoe_normalized * (GEODECT_B_MAX - GEODECT_B_MIN)
    )

    lab_image[~inside] = 0.0

    rgb_float = lab_to_rgb(
        lab_image[..., 0],
        lab_image[..., 1],
        lab_image[..., 2],
    )

    colour_image = np.clip(
        rgb_float * 255.0,
        0.0,
        255.0
    ).astype(
        np.uint8
    )

    rgb = np.empty(
        zeff.shape + (3,),
        dtype=np.uint8
    )

    rgb[:] = np.asarray(
        background_rgb,
        dtype=np.uint8
    )

    rgb[inside] = colour_image[inside]

    rgb[invalid] = np.asarray(
        background_rgb,
        dtype=np.uint8
    )

    return rgb

def create_zeff_colorbar(
    zeff_min,
    zeff_max,
    width=256,
    height=32
):
    """
    Create a horizontal RGB colorbar for Zeff.

    The returned image contains only the colour gradient.
    Numeric labels will be added in the GUI.
    """

    zeff_min = float(zeff_min)
    zeff_max = float(zeff_max)

    if zeff_max <= zeff_min:
        raise ValueError(
            "zeff_max must be greater than zeff_min."
        )

    values = np.linspace(
        zeff_min,
        zeff_max,
        width,
        dtype=np.float32
    )

    values = np.tile(
        values[None, :],
        (height, 1)
    )

    return create_zeff_rgb(
        values,
        zeff_min,
        zeff_max,
    )


def create_rhoe_colorbar(
    rhoe_min,
    rhoe_max,
    width=256,
    height=32
):
    """
    Create a horizontal RGB colorbar for rho_e.

    The returned image contains only the colour gradient.
    Numeric labels will be added in the GUI.
    """

    rhoe_min = float(rhoe_min)
    rhoe_max = float(rhoe_max)

    if rhoe_max <= rhoe_min:
        raise ValueError(
            "rhoe_max must be greater than rhoe_min."
        )

    values = np.linspace(
        rhoe_min,
        rhoe_max,
        width,
        dtype=np.float32
    )

    values = np.tile(
        values[None, :],
        (height, 1)
    )

    return create_rhoe_rgb(
        values,
        rhoe_min,
        rhoe_max,
    )


def create_geodect_legend(
    zeff_min,
    zeff_max,
    rhoe_min,
    rhoe_max,
    width=256,
    height=256
):
    """
    Create the two-dimensional Geo_DECT colour legend.

    Horizontal axis:
        Zeff, increasing from left to right.

    Vertical axis:
        rho_e, increasing from bottom to top.

    The returned image contains only the colour field.
    Axis ticks and numeric labels will be added in the GUI.
    """

    zeff_min = float(zeff_min)
    zeff_max = float(zeff_max)
    rhoe_min = float(rhoe_min)
    rhoe_max = float(rhoe_max)

    if zeff_max <= zeff_min:
        raise ValueError(
            "zeff_max must be greater than zeff_min."
        )

    if rhoe_max <= rhoe_min:
        raise ValueError(
            "rhoe_max must be greater than rhoe_min."
        )

    zeff_values = np.linspace(
        zeff_min,
        zeff_max,
        width,
        dtype=np.float32
    )

    rhoe_values = np.linspace(
        rhoe_max,
        rhoe_min,
        height,
        dtype=np.float32
    )

    zeff_grid = np.tile(
        zeff_values[None, :],
        (height, 1)
    )

    rhoe_grid = np.tile(
        rhoe_values[:, None],
        (1, width)
    )

    return create_geodect_rgb(
        zeff_grid,
        rhoe_grid,
        zeff_min,
        zeff_max,
        rhoe_min,
        rhoe_max,
    )


def create_tick_values(
    value_min,
    value_max,
    tick_count=5
):
    """
    Return evenly spaced numeric tick values for a legend.
    """

    value_min = float(value_min)
    value_max = float(value_max)
    tick_count = int(tick_count)

    if value_max <= value_min:
        raise ValueError(
            "value_max must be greater than value_min."
        )

    if tick_count < 2:
        raise ValueError(
            "tick_count must be at least 2."
        )

    return np.linspace(
        value_min,
        value_max,
        tick_count,
        dtype=np.float32
    )