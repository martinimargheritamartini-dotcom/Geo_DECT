import numpy as np
from scipy import ndimage


WATER_ELECTRON_DENSITY_CM3 = 3.343e23


def compute_hu(stored_pixel, rescale_slope=1.0, rescale_intercept=0.0):
    """Convert stored DICOM pixel values into Hounsfield units."""
    stored_pixel = np.asarray(stored_pixel, dtype=np.float32)
    hu = (
        stored_pixel * float(rescale_slope)
        + float(rescale_intercept)
    )
    return hu.astype(np.float32, copy=False)


def compute_mu_from_hu(hu_values):
    """Convert HU into attenuation relative to water."""
    hu_values = np.asarray(hu_values, dtype=np.float32)
    return (hu_values / 1000.0 + 1.0).astype(
        np.float32,
        copy=False,
    )


def compute_mu(gray_level, calibration):
    """
    Convert TIFF gray levels or stored DICOM pixels into mu.

    Supported modes:
      - slope_offset
      - physical_min_max
      - dicom_hu
    """
    if not isinstance(calibration, dict):
        raise TypeError("calibration must be a dictionary.")

    mode = calibration.get("mode")
    if mode is None:
        raise ValueError(
            "The calibration dictionary must contain a 'mode' field."
        )

    gray_level = np.asarray(gray_level, dtype=np.float32)

    if mode == "slope_offset":
        if "slope" not in calibration or "offset" not in calibration:
            raise ValueError(
                "Slope/offset calibration requires 'slope' and 'offset'."
            )
        mu = (
            gray_level * float(calibration["slope"])
            + float(calibration["offset"])
        )

    elif mode == "physical_min_max":
        if "min" not in calibration or "max" not in calibration:
            raise ValueError(
                "Physical-range calibration requires 'min' and 'max'."
            )

        physical_min = float(calibration["min"])
        physical_max = float(calibration["max"])
        stored_max = float(calibration.get("stored_max", 65535.0))

        if physical_max <= physical_min:
            raise ValueError(
                "Physical maximum must be greater than physical minimum."
            )
        if stored_max <= 0:
            raise ValueError("stored_max must be greater than zero.")

        mu = (
            gray_level / stored_max
        ) * (
            physical_max - physical_min
        ) + physical_min

    elif mode == "dicom_hu":
        # In this mode gray_level is already expressed in HU.
        mu = compute_mu_from_hu(gray_level)

    else:
        raise ValueError(f"Unsupported calibration mode: {mode}")

    return mu.astype(np.float32, copy=False)


def evaluate_polynomial(x, coefficients):
    """Evaluate a polynomial with coefficients ordered by power."""
    if coefficients is None or len(coefficients) == 0:
        raise ValueError("At least one coefficient is required.")

    x = np.asarray(x, dtype=np.float32)
    result = np.zeros_like(x, dtype=np.float32)

    for power, coefficient in enumerate(coefficients):
        result += np.float32(coefficient) * np.power(x, power)

    return result.astype(np.float32, copy=False)


def compute_zeff(
    mu_low, mu_high, coefficients, valid_mask=None,
    plausible_min=0.0, plausible_max=50.0,
    ratio_min=None, ratio_max=None,
):
    """Calculate Zeff from mu_LOW / mu_HIGH.

    Voxels whose mu_low/mu_high ratio falls far outside the range the
    calibration curve was actually fitted on can extrapolate to
    physically meaningless results (a negative effective atomic
    number, or one far beyond any real material) -- and because rho_e
    is computed FROM this Zeff, an implausible Zeff here silently
    propagates into wildly wrong rho_e values downstream (this is what
    was crashing Dragonfly's channel creation on some datasets: values
    like -35 for Zeff and ~1e26 for rho_e were being written straight
    into the volume). Any voxel whose computed Zeff falls outside
    [plausible_min, plausible_max] is treated as invalid (NaN) here,
    the same way a non-finite or non-positive mu already is, rather
    than being passed downstream as a real result.
    """
    mu_low = np.asarray(mu_low, dtype=np.float32)
    mu_high = np.asarray(mu_high, dtype=np.float32)

    if mu_low.shape != mu_high.shape:
        raise ValueError(
            "mu_low and mu_high must have the same shape."
        )

    valid = (
        np.isfinite(mu_low)
        & np.isfinite(mu_high)
        & (mu_low > 0)
        & (mu_high > 0)
    )

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != mu_low.shape:
            raise ValueError(
                "valid_mask must have the same shape as the input arrays."
            )
        valid &= valid_mask

    zeff = np.full_like(mu_low, np.nan, dtype=np.float32)

    if np.any(valid):
        ratio = mu_low[valid] / mu_high[valid]
        computed = evaluate_polynomial(ratio, coefficients)
        # Optional mu_LOW/mu_HIGH plausible range (None = disabled):
        # voxels whose ratio lies outside [ratio_min, ratio_max] are
        # excluded before their Zeff is accepted.
        ratio_check = ratio.astype(np.float64)
        ratio_out = np.zeros(ratio.shape, dtype=bool)
        if ratio_min is not None:
            ratio_out |= ratio_check < float(ratio_min)
        if ratio_max is not None:
            ratio_out |= ratio_check > float(ratio_max)
        computed[ratio_out] = np.nan
        # Compare in float64 explicitly: computed is float32, and a
        # bare "computed > plausible_max" lets NumPy round the plain
        # Python float boundary down to the nearest float32 value --
        # for large boundaries that rounding can go the WRONG way
        # (up), silently admitting values above the intended cap.
        # Casting computed to float64 for the comparison only keeps
        # the enforced boundary exact regardless of magnitude.
        computed_check = computed.astype(np.float64)
        implausible = (
            ~np.isfinite(computed)
            | (computed_check < plausible_min)
            | (computed_check > plausible_max)
        )
        computed[implausible] = np.nan
        zeff[valid] = computed

    return zeff


def compute_rhoe(
    mu_low,
    mu_high,
    zeff,
    coefficients,
    numerator_mode="low",
    valid_mask=None,
    plausible_min=0.0,
    plausible_max=1e24,
):
    """
    Calculate electron density from the selected mu numerator.

    The returned convention follows the calibration coefficients:
    industrial TIFF calibrations commonly return absolute rho_e,
    while medical DICOM calibrations commonly return rho_e/rho_e,water.
    plausible_max is deliberately generous (1e24, ~10x above the
    largest real absolute rho_e ever seen in practice) so it does not
    interfere with either convention -- it exists only to catch the
    same kind of extrapolation blow-up compute_zeff guards against
    (division by a calibration curve evaluated somewhere it was never
    fitted can produce something like 1e26, ~1000x too large, which is
    what was crashing Dragonfly's channel creation on some datasets).
    """
    mu_low = np.asarray(mu_low, dtype=np.float32)
    mu_high = np.asarray(mu_high, dtype=np.float32)
    zeff = np.asarray(zeff, dtype=np.float32)

    if (
        mu_low.shape != mu_high.shape
        or mu_low.shape != zeff.shape
    ):
        raise ValueError(
            "mu_low, mu_high and zeff must have the same shape."
        )

    if numerator_mode == "low":
        numerator = mu_low
    elif numerator_mode == "high":
        numerator = mu_high
    elif numerator_mode == "mean":
        numerator = (mu_low + mu_high) / 2.0
    else:
        raise ValueError(
            "numerator_mode must be 'low', 'high' or 'mean'."
        )

    denominator = evaluate_polynomial(zeff, coefficients)

    valid = (
        np.isfinite(mu_low)
        & np.isfinite(mu_high)
        & (mu_low > 0)
        & (mu_high > 0)
        & np.isfinite(zeff)
        & np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-26)
    )

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != zeff.shape:
            raise ValueError(
                "valid_mask must have the same shape as the input arrays."
            )
        valid &= valid_mask

    rhoe = np.full_like(zeff, np.nan, dtype=np.float32)

    if np.any(valid):
        computed = numerator[valid] / denominator[valid]
        # Same float64-comparison fix as compute_zeff above: comparing
        # a float32 array directly against a large plain-Python-float
        # boundary like 1e24 can have NumPy round THAT boundary to the
        # nearest float32 value first -- for 1e24 specifically that
        # rounds UP to about 1.0000000138e24, letting values genuinely
        # above the intended cap slip through untouched. This is what
        # was crashing Dragonfly's channel creation again: a large
        # fraction of voxels sitting exactly at that rounded-up
        # boundary instead of being caught as implausible.
        computed_check = computed.astype(np.float64)
        implausible = (
            ~np.isfinite(computed)
            | (computed_check < plausible_min)
            | (computed_check > plausible_max)
        )
        computed[implausible] = np.nan
        rhoe[valid] = computed

    return rhoe


def relative_to_absolute_rhoe(
    rhoe_relative,
    water_electron_density=WATER_ELECTRON_DENSITY_CM3,
):
    """Convert rho_e/rho_e,water into absolute rho_e in electrons/cm^3."""
    rhoe_relative = np.asarray(rhoe_relative, dtype=np.float32)
    water_electron_density = float(water_electron_density)

    if water_electron_density <= 0:
        raise ValueError(
            "water_electron_density must be greater than zero."
        )

    return (
        rhoe_relative * water_electron_density
    ).astype(np.float32, copy=False)

def denoise_mu(mu, method="none", strength=1.5, valid_mask=None):
    """Reduce noise in one 2D mu slice before the mu_LOW/mu_HIGH ratio.

    method:
      - "none":     returned unchanged
      - "gaussian": mask-aware Gaussian (normalized convolution), sigma =
                    strength in pixels; voxels outside valid_mask never
                    contribute, so the background does not bleed into the
                    sample edge
      - "median":   median filter, window = strength rounded to an odd
                    integer >= 3; outside valid_mask the input is kept
    Only voxels inside valid_mask are modified.
    """
    mu = np.asarray(mu, dtype=np.float32)
    if method in (None, "", "none"):
        return mu

    if valid_mask is None:
        mask = np.isfinite(mu)
    else:
        mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(mu)
    if not np.any(mask):
        return mu

    out = mu.copy()

    if method == "gaussian":
        sigma = max(float(strength), 1e-6)
        weights = mask.astype(np.float64)
        filled = np.where(mask, mu, 0.0).astype(np.float64)
        num = ndimage.gaussian_filter(filled, sigma=sigma, mode="nearest")
        den = ndimage.gaussian_filter(weights, sigma=sigma, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = num / den
        good = mask & (den > 1e-6)
        out[good] = smoothed[good].astype(np.float32)
        return out

    if method == "median":
        size = max(3, int(round(float(strength))))
        if size % 2 == 0:
            size += 1
        # Fill the background with the median of the retained voxels so
        # that it cannot pull edge values toward zero.
        fill_value = float(np.median(mu[mask]))
        filled = np.where(mask, mu, fill_value).astype(np.float32)
        smoothed = ndimage.median_filter(filled, size=size, mode="nearest")
        out[mask] = smoothed[mask]
        return out

    raise ValueError("method must be 'none', 'gaussian' or 'median'.")
