"""
Statistics and PDF report generation for the Geo_DECT plugin.

Kept in its own module (separate from geodect_3d.py, which talks to the
Dragonfly API) because everything here only needs NumPy + Matplotlib and
works on plain arrays -- easier to keep correct and to re-use for both
the per-property (Z_eff / rho_e) reports and the combined crossplot.

Unit handling
-------------
Every function in this module expects values ALREADY IN THEIR TRUE
PHYSICAL UNITS (Z_eff dimensionless, rho_e in electrons/cm^3 -- NOT the
1e23-divided "display" numbers used elsewhere in the plugin to keep the
3D legend readable). The caller is responsible for multiplying rho_e
back by its display scale before calling anything here. This keeps this
module simple and unambiguous: numbers in, correct numbers out.
"""

import numpy as np


# Rendering (boxplot fliers, scatter/hexbin points, k-means) on the full
# voxel count (which can be hundreds of millions for a 3D volume) is what
# was freezing Dragonfly -- matplotlib ends up trying to draw one marker
# per outlier point, and the k-means below computes a full (n, k)
# distance matrix per iteration. Everything that touches the FULL array
# stays limited to fast vectorized NumPy calls (mean/median/percentile/
# histogram counts); anything that gets individually plotted or looped
# over is capped to a random subsample of this size.
_MAX_RENDERED_POINTS = 200_000


def _subsample(*arrays, max_points=_MAX_RENDERED_POINTS, seed=0):
    """Return a random subsample (same indices across all arrays) capped
    at max_points, without ever needing to sort or copy the full data."""
    size = arrays[0].shape[0]
    if size <= max_points:
        return arrays if len(arrays) > 1 else arrays[0]
    rng = np.random.default_rng(seed)
    indices = rng.choice(size, size=max_points, replace=False)
    subsampled = tuple(array[indices] for array in arrays)
    return subsampled if len(arrays) > 1 else subsampled[0]


def _gradient_keep_mask(slice_values, valid, keep, exclude_percent):
    """Drop, among the voxels in ``keep``, the given percentage with the
    strongest local (Sobel) gradient -- the voxels on phase boundaries."""
    if exclude_percent <= 0.0 or not np.any(keep):
        return keep
    from scipy import ndimage

    filled = np.where(valid, slice_values, 0.0).astype(np.float32)
    gradient = np.hypot(
        ndimage.sobel(filled, axis=0),
        ndimage.sobel(filled, axis=1),
    )
    threshold = np.percentile(gradient[keep], 100.0 - exclude_percent)
    return keep & (gradient <= threshold)


def select_interior_pairs(volume_a, volume_b, exclude_gradient_percent=0.0):
    """Paired version of select_interior_values for the crossplot: a voxel
    is kept only if it is valid (> 0) in BOTH volumes and, when
    requested, is not among the strongest-gradient voxels of EITHER
    volume (same per-slice criterion as select_interior_values).

    Returns (values_a, values_b, total_valid_count, excluded_count),
    float32 1D arrays in the units the volumes are stored in.
    """
    exclude_gradient_percent = float(exclude_gradient_percent)
    chunks_a, chunks_b = [], []
    total_valid = 0
    kept_total = 0
    for z_index in range(volume_a.shape[0]):
        slice_a = np.asarray(volume_a[z_index], dtype=np.float32)
        slice_b = np.asarray(volume_b[z_index], dtype=np.float32)
        valid_a = np.isfinite(slice_a) & (slice_a > 0.0)
        valid_b = np.isfinite(slice_b) & (slice_b > 0.0)
        valid = valid_a & valid_b
        count = int(np.count_nonzero(valid))
        if count == 0:
            continue
        total_valid += count
        keep = _gradient_keep_mask(
            slice_a, valid_a, valid, exclude_gradient_percent
        )
        keep = _gradient_keep_mask(
            slice_b, valid_b, keep, exclude_gradient_percent
        )
        kept_total += int(np.count_nonzero(keep))
        if np.any(keep):
            chunks_a.append(slice_a[keep])
            chunks_b.append(slice_b[keep])
    if not chunks_a:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, total_valid, total_valid
    return (
        np.concatenate(chunks_a),
        np.concatenate(chunks_b),
        total_valid,
        total_valid - kept_total,
    )


def select_interior_values(
    volume,
    scale=1.0,
    exclude_gradient_percent=0.0,
    edge_erosion_px=0,
):
    """Collect the valid (> 0) voxel values of a (z, y, x) volume,
    optionally excluding interface (partial-volume) voxels.

    Partial-volume voxels at the boundary between two phases (e.g. an
    ooid cortex and the surrounding matrix) carry intermediate values
    that fill the gap between the two real populations of a histogram.
    Two standard, explicitly reported criteria are offered here, both
    applied slice by slice:

    - edge_erosion_px: drop voxels within this many pixels of the
      sample boundary / any excluded region (outer surface, pores).
    - exclude_gradient_percent: inside each slice, drop the given
      percentage of valid voxels with the strongest local gradient
      (Sobel magnitude) -- i.e. the voxels sitting on internal phase
      boundaries. Homogeneous interiors of each phase are kept.

    Returns (values_float32_1d, total_valid_count, excluded_count).
    Values are multiplied by ``scale`` (rho_e display scale).
    """
    from scipy import ndimage

    exclude_gradient_percent = float(exclude_gradient_percent)
    edge_erosion_px = int(edge_erosion_px)
    chunks = []
    total_valid = 0
    kept_total = 0

    for z_index in range(volume.shape[0]):
        slice_values = np.asarray(volume[z_index], dtype=np.float32)
        valid = np.isfinite(slice_values) & (slice_values > 0.0)
        count = int(np.count_nonzero(valid))
        if count == 0:
            continue
        total_valid += count
        keep = valid

        if edge_erosion_px > 0:
            keep = ndimage.binary_erosion(
                keep, iterations=edge_erosion_px
            )

        keep = _gradient_keep_mask(
            slice_values, valid, keep, exclude_gradient_percent
        )

        selected = slice_values[keep]
        kept_total += int(selected.size)
        if selected.size:
            chunks.append(selected)

    if not chunks:
        return np.zeros(0, dtype=np.float32), total_valid, total_valid

    values = np.concatenate(chunks)
    if scale != 1.0:
        values = values.astype(np.float64) * float(scale)
    return values, total_valid, total_valid - kept_total


def fit_gaussian_mixture_1d(
    values, n_components=None, max_components=3, seed=0,
):
    """Fit a 1D Gaussian mixture with EM (NumPy only).

    n_components=None selects the number of populations (1 to
    max_components) with the Bayesian Information Criterion (lowest
    BIC wins). Fitting runs on a random subsample (see _subsample), which
    gives the same component parameters as the full volume for all
    practical purposes.

    Returns a dict:
      components: list of dicts (weight, mean, std), sorted by mean
      n_components: chosen number of components
      bic: {k: BIC} for every k tried
      n_fit: number of values used for the fit
    """
    data = np.asarray(
        _subsample(np.asarray(values), max_points=100_000), dtype=np.float64
    )
    data = data[np.isfinite(data)]
    if data.size < 50:
        raise ValueError("Too few values for a mixture fit.")

    candidates = (
        [int(n_components)] if n_components
        else list(range(1, int(max_components) + 1))
    )

    def _fit(k):
        quantiles = np.linspace(0, 100, k + 2)[1:-1]
        means = np.percentile(data, quantiles)
        spread = np.std(data)
        stds = np.full(k, spread / k)
        weights = np.full(k, 1.0 / k)
        previous = -np.inf
        log_likelihood = -np.inf
        for _ in range(500):
            stds = np.maximum(stds, 1e-6 * max(spread, 1e-12))
            log_density = (
                -0.5 * ((data[:, None] - means) / stds) ** 2
                - np.log(stds * np.sqrt(2.0 * np.pi))
                + np.log(np.maximum(weights, 1e-300))
            )
            peak = log_density.max(axis=1, keepdims=True)
            log_total = peak[:, 0] + np.log(
                np.exp(log_density - peak).sum(axis=1)
            )
            log_likelihood = float(log_total.sum())
            responsibilities = np.exp(log_density - log_total[:, None])
            totals = responsibilities.sum(axis=0) + 1e-12
            weights = totals / data.size
            means = (responsibilities * data[:, None]).sum(axis=0) / totals
            stds = np.sqrt(
                (responsibilities * (data[:, None] - means) ** 2).sum(axis=0)
                / totals
            )
            if abs(log_likelihood - previous) < 1e-6 * abs(log_likelihood):
                break
            previous = log_likelihood
        parameter_count = 3 * k - 1
        bic = -2.0 * log_likelihood + parameter_count * np.log(data.size)
        order = np.argsort(means)
        components = [
            {
                "weight": float(weights[i]),
                "mean": float(means[i]),
                "std": float(stds[i]),
            }
            for i in order
        ]
        return components, float(bic)

    results = {k: _fit(k) for k in candidates}
    bics = {k: result[1] for k, result in results.items()}
    best = min(bics, key=bics.get)
    return {
        "components": results[best][0],
        "n_components": best,
        "bic": bics,
        "n_fit": int(data.size),
    }


def fit_gaussian_mixture_2d(
    points, n_components=None, max_components=4, seed=0,
):
    """Fit a 2D Gaussian mixture (full covariance) with EM, NumPy only.

    Both variables are standardised (zero mean, unit variance) before
    fitting and the results are transformed back to data units, so
    Z_eff and rho_e weigh equally regardless of their very different
    numerical scales. The number of populations is chosen by BIC when
    n_components is None.

    Returns a dict like fit_gaussian_mixture_1d, each component holding
    weight, mean (2,), cov (2, 2) in DATA units, sorted by Z_eff mean.
    """
    data = np.asarray(
        _subsample(np.asarray(points, dtype=np.float64), max_points=100_000),
        dtype=np.float64,
    )
    data = data[np.all(np.isfinite(data), axis=1)]
    if data.shape[0] < 100:
        raise ValueError("Too few points for a 2D mixture fit.")

    centre = data.mean(axis=0)
    spread = data.std(axis=0)
    spread[spread == 0] = 1.0
    z = (data - centre) / spread
    n = z.shape[0]
    rng = np.random.default_rng(seed)

    candidates = (
        [int(n_components)] if n_components
        else list(range(1, int(max_components) + 1))
    )

    def _log_gauss(x, mean, cov):
        cov = cov + np.eye(2) * 1e-6
        inv = np.linalg.inv(cov)
        diff = x - mean
        maha = np.einsum("ni,ij,nj->n", diff, inv, diff)
        return -0.5 * maha - 0.5 * np.log(np.linalg.det(cov)) - np.log(2 * np.pi)

    def _fit(k):
        best = None
        for attempt in range(3):
            # k-means++-like initialisation on the standardised data
            means = [z[rng.integers(n)]]
            for _ in range(1, k):
                dist = np.min(
                    [((z - m) ** 2).sum(axis=1) for m in means], axis=0
                )
                means.append(z[rng.choice(n, p=dist / dist.sum())])
            means = np.array(means)
            covs = np.array([np.eye(2) * 0.5 for _ in range(k)])
            weights = np.full(k, 1.0 / k)
            previous = -np.inf
            log_likelihood = -np.inf
            for _ in range(300):
                log_density = np.column_stack([
                    np.log(max(weights[j], 1e-300))
                    + _log_gauss(z, means[j], covs[j])
                    for j in range(k)
                ])
                peak = log_density.max(axis=1, keepdims=True)
                log_total = peak[:, 0] + np.log(
                    np.exp(log_density - peak).sum(axis=1)
                )
                log_likelihood = float(log_total.sum())
                resp = np.exp(log_density - log_total[:, None])
                totals = resp.sum(axis=0) + 1e-12
                weights = totals / n
                means = (resp.T @ z) / totals[:, None]
                for j in range(k):
                    diff = z - means[j]
                    covs[j] = (resp[:, j, None] * diff).T @ diff / totals[j]
                if abs(log_likelihood - previous) < 1e-6 * abs(log_likelihood):
                    break
                previous = log_likelihood
            if best is None or log_likelihood > best[0]:
                best = (log_likelihood, weights.copy(), means.copy(), covs.copy())
        log_likelihood, weights, means, covs = best
        parameter_count = 6 * k - 1
        bic = -2.0 * log_likelihood + parameter_count * np.log(n)
        components = []
        scale = np.diag(spread)
        for j in np.argsort(means[:, 0]):
            components.append({
                "weight": float(weights[j]),
                "mean": means[j] * spread + centre,
                "cov": scale @ covs[j] @ scale,
            })
        return components, float(bic)

    results = {k: _fit(k) for k in candidates}
    bics = {k: result[1] for k, result in results.items()}
    best_k = min(bics, key=bics.get)
    return {
        "components": results[best_k][0],
        "n_components": best_k,
        "bic": bics,
        "n_fit": int(n),
    }


def compute_scalar_statistics(values, selected_minimum, selected_maximum):
    """
    Compute summary statistics for one property (Z_eff or rho_e).

    ``values`` must already be filtered to valid, finite, physically
    meaningful voxels (e.g. > 0, background excluded) and in true
    physical units.

    Every reported statistic (mean, median, mode, std, min, max,
    percentiles) is computed ONLY from voxels inside the selected
    range -- not the whole sample. The whole-sample mean/std are kept
    separately (under "whole_sample_mean"/"whole_sample_std") purely as
    the reference curve showing the overall trend; they are not
    reported as "the" statistics.

    Returns a dict of plain Python floats/ints, ready to format for
    display or to hand to create_scalar_statistics_figure().
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    total = int(values.size)
    if total == 0:
        raise ValueError("No valid values to compute statistics from.")

    selected_minimum = float(selected_minimum)
    selected_maximum = float(selected_maximum)

    whole_sample_mean = float(np.mean(values))
    whole_sample_std = float(np.std(values))

    below = values < selected_minimum
    above = values > selected_maximum
    inside = (~below) & (~above)

    below_count = int(np.count_nonzero(below))
    above_count = int(np.count_nonzero(above))
    inside_count = int(np.count_nonzero(inside))

    below_values = values[below]
    above_values = values[above]
    inside_values = values[inside]

    if inside_count == 0:
        raise ValueError(
            "No voxels fall inside the selected range -- nothing to "
            "compute statistics from."
        )

    mean = float(np.mean(inside_values))
    median = float(np.median(inside_values))
    std = float(np.std(inside_values))
    data_min = float(np.min(inside_values))
    data_max = float(np.max(inside_values))

    # A continuous property has no exact mode -- approximate it as the
    # centre of the tallest bin of a fine histogram over the
    # inside-range values only.
    hist_counts, hist_edges = np.histogram(inside_values, bins=200)
    peak_index = int(np.argmax(hist_counts))
    mode = float(
        (hist_edges[peak_index] + hist_edges[peak_index + 1]) / 2.0
    )

    percentiles = {
        p: float(np.percentile(inside_values, p))
        for p in (1, 5, 25, 50, 75, 95, 99)
    }

    return {
        "total": total,
        "mean": mean,
        "median": median,
        "mode": mode,
        "std": std,
        "min": data_min,
        "max": data_max,
        "percentiles": percentiles,
        "selected_minimum": selected_minimum,
        "selected_maximum": selected_maximum,
        "inside_count": inside_count,
        "below_count": below_count,
        "above_count": above_count,
        "inside_fraction": inside_count / total,
        "below_fraction": below_count / total,
        "above_fraction": above_count / total,
        "below_min": float(below_values.min()) if below_count else None,
        "below_max": float(below_values.max()) if below_count else None,
        "above_min": float(above_values.min()) if above_count else None,
        "above_max": float(above_values.max()) if above_count else None,
        "whole_sample_mean": whole_sample_mean,
        "whole_sample_std": whole_sample_std,
    }


def _format_value(value, unit_suffix=""):
    """Format one physical value with a fixed, unambiguous notation.

    Uses plain decimal for "normal-sized" numbers and explicit scientific
    notation (e.g. 1.234e+23) for very large/small ones -- never a
    silently-divided number, so the printed figure is always the true
    physical value.
    """
    if value is None:
        return "n/a"
    magnitude = abs(value)
    if magnitude != 0 and (magnitude < 1e-3 or magnitude >= 1e5):
        text = "{:.4e}".format(value)
    else:
        text = "{:.4f}".format(value)
    return text + unit_suffix


def create_scalar_statistics_figure(
    values,
    selected_minimum,
    selected_maximum,
    property_label,
    unit_suffix="",
    color="tab:blue",
    n_components=None,
    exclusion_note="",
):
    """
    Build a Matplotlib Figure with a distribution histogram + Gaussian
    mixture fit (left) and a boxplot + percentiles (right), plus a summary
    statistics text box -- for one scalar property (Z_eff or rho_e).

    ``property_label`` should already be a Matplotlib mathtext string
    for correct typesetting, e.g. r"$Z_{eff}$" or r"$\\rho_e$".
    ``values``, ``selected_minimum``, ``selected_maximum`` must be in
    true physical units (see module docstring).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    selected_minimum = float(selected_minimum)
    selected_maximum = float(selected_maximum)

    stats = compute_scalar_statistics(
        values, selected_minimum, selected_maximum
    )

    def _fmt(value):
        """Format a value WITH its unit, so every number in the report
        is unambiguous (e.g. rho_e shown in units of 10^23 e-/cm3)."""
        text = _format_value(value)
        if unit_suffix and value is not None:
            text += " " + unit_suffix
        return text

    inside_mask = (
        (values >= selected_minimum) & (values <= selected_maximum)
    )
    inside_values = values[inside_mask]

    # The true min/max of a real dataset is often pulled far out by a
    # handful of outlier voxels (a stray artifact, background noise...).
    # Scaling the plot to that true extreme squashes the actual
    # distribution into an unreadable sliver. Zoom instead to a padded
    # window around the selected range and the P1-P99 spread -- wide
    # enough to show the real shape, without outliers dominating the
    # axis. The true min/max stay fully reported in the text box below.
    display_low = min(
        selected_minimum, stats["percentiles"][1], stats["mean"]
    )
    display_high = max(
        selected_maximum, stats["percentiles"][99], stats["mean"]
    )
    display_span = max(display_high - display_low, 1e-9)
    display_low -= 0.08 * display_span
    display_high += 0.08 * display_span

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        2, 2, height_ratios=(5, 2.2), width_ratios=(1, 1),
        left=0.06, right=0.78, top=0.94, bottom=0.04,
        wspace=0.85, hspace=0.35,
    )
    ax_hist = fig.add_subplot(grid[0, 0])
    ax_box = fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[1, :])
    ax_text.axis("off")

    # -- Left: histogram of the selected (inside-range) values -- all
    # the numeric statistics below (mean, median, mode, percentiles,
    # the boxplot) are computed from these SAME inside-range values,
    # not the whole sample. The shaded curves are a Gaussian mixture
    # fitted on these same values (one curve per population).
    if inside_values.size:
        ax_hist.hist(
            inside_values,
            bins=100,
            density=True,
            color="lightgray",
            edgecolor="gray",
            linewidth=0.5,
            label="Inside-range values",
        )

    # Gaussian mixture fitted on the inside-range values: one curve per
    # population (scaled by its volume fraction) plus their sum. The
    # number of populations is chosen by BIC unless forced by the caller.
    mixture = None
    if inside_values.size >= 50:
        try:
            mixture = fit_gaussian_mixture_1d(
                inside_values, n_components=n_components
            )
        except Exception:
            mixture = None
    stats["mixture"] = mixture

    if mixture is not None:
        curve_x = np.linspace(display_low, display_high, 600)
        total_curve = np.zeros_like(curve_x)
        population_colors = ("tab:blue", "tab:orange", "tab:green")
        for index, component in enumerate(mixture["components"]):
            curve = component["weight"] * (
                1.0 / (component["std"] * np.sqrt(2.0 * np.pi))
            ) * np.exp(
                -0.5 * ((curve_x - component["mean"]) / component["std"]) ** 2
            )
            total_curve += curve
            ax_hist.fill_between(
                curve_x, curve,
                color=population_colors[index % len(population_colors)],
                alpha=0.30,
                label="Population {}: {} \u00b1 {}{} ({:.0f}%)".format(
                    index + 1,
                    _format_value(component["mean"]),
                    _format_value(component["std"]),
                    " " + unit_suffix if unit_suffix else "",
                    component["weight"] * 100.0,
                ),
            )
        ax_hist.plot(
            curve_x, total_curve, color="black", linewidth=1.5,
            label="Mixture model (sum)",
        )

    ax_hist.axvline(stats["mean"], color="red", linestyle="--", linewidth=1.5)
    ax_hist.axvline(
        stats["median"], color="blue", linestyle="-.", linewidth=1.5
    )
    ax_hist.axvline(
        stats["mode"], color="green", linestyle=":", linewidth=1.5
    )
    ax_hist.axvline(
        selected_minimum, color="purple", linestyle="--", linewidth=1.5
    )
    ax_hist.axvline(
        selected_maximum, color="darkorange", linestyle="--", linewidth=1.5
    )
    ax_hist.set_xlim(display_low, display_high)

    ax_hist.set_title("Distribution of {} values".format(property_label))
    ax_hist.set_xlabel(property_label + (
        " ({})".format(unit_suffix) if unit_suffix else ""
    ))
    ax_hist.set_ylabel("Probability density")
    ax_hist.legend(loc="upper left", fontsize=7.5, framealpha=0.85)

    annotation_lines = [
        ("Mean: " + _fmt(stats["mean"]), "red"),
        ("Median: " + _fmt(stats["median"]), "blue"),
        ("Mode: " + _fmt(stats["mode"]), "green"),
        ("Selected min: " + _fmt(selected_minimum), "purple"),
        ("Selected max: " + _fmt(selected_maximum), "darkorange"),
    ]
    for index, (text, text_color) in enumerate(annotation_lines):
        ax_hist.annotate(
            text,
            xy=(1.03, 0.92 - index * 0.13),
            xycoords="axes fraction",
            color=text_color,
            fontsize=10,
            fontweight="bold",
        )

    # -- Right: boxplot of the INSIDE-RANGE values only (same values
    # the mean/median/percentiles above are computed from), with
    # percentile labels. Fliers (individual outlier markers) are
    # disabled and the data is subsampled: rendering one marker per
    # outlier voxel (there can be hundreds of millions of them) is what
    # was freezing Dragonfly, and the percentile annotations already
    # convey the same information.
    boxplot_values = _subsample(inside_values)
    box = ax_box.boxplot(
        boxplot_values,
        vert=True,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "red", "linewidth": 2},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("lightgray")

    ax_box.set_title("{} — boxplot and percentiles".format(property_label))
    ax_box.set_ylabel(property_label + (
        " ({})".format(unit_suffix) if unit_suffix else ""
    ))
    ax_box.set_xticks([])
    # Same zoomed window as the histogram, so both panels read the same
    # scale. Min/Max here are the extremes of the inside-range values
    # (see compute_scalar_statistics), so they normally sit close to
    # the selected range's own edges.
    ax_box.set_ylim(display_low, display_high)
    ax_box.set_xlim(0.4, 2.6)

    percentile_rows = [
        ("Max", stats["max"], "black"),
        ("P99", stats["percentiles"][99], "dimgray"),
        ("P95", stats["percentiles"][95], "dimgray"),
        ("P75", stats["percentiles"][75], "tab:blue"),
        ("P50 / Median", stats["percentiles"][50], "red"),
        ("P25", stats["percentiles"][25], "tab:blue"),
        ("P5", stats["percentiles"][5], "dimgray"),
        ("P1", stats["percentiles"][1], "dimgray"),
        ("Min", stats["min"], "black"),
    ]
    # Labels are placed at fixed, EVENLY SPACED rows (top to bottom) with
    # a thin leader line pointing back to the actual data value -- rather
    # than at a position proportional to the value itself, which is what
    # made every row pile on top of each other when several percentiles
    # sit close together (very common with real, skewed data).
    row_count = len(percentile_rows)
    row_ys = np.linspace(0.94, 0.06, row_count)
    for (label, value, row_color), row_y in zip(percentile_rows, row_ys):
        clipped_value = min(max(value, display_low), display_high)
        ax_box.annotate(
            "{}: {}".format(label, _fmt(value)),
            xy=(1.0, clipped_value),
            xycoords=("axes fraction", "data"),
            xytext=(1.2, row_y),
            textcoords="axes fraction",
            color=row_color,
            fontsize=9,
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": row_color,
                "linewidth": 0.7,
                "alpha": 0.6,
                "shrinkA": 0, "shrinkB": 0,
            },
        )

    # -- Summary text box.
    summary_lines = [
        "SUMMARY STATISTICS — {}{}".format(
            property_label,
            " ({})".format(unit_suffix) if unit_suffix else "",
        ),
        "Mean: {}    Median: {}    Mode: {}    Std. dev.: {}".format(
            _fmt(stats["mean"]),
            _fmt(stats["median"]),
            _fmt(stats["mode"]),
            _fmt(stats["std"]),
        ),
        "",
        "Min: {}    Max: {}".format(
            _fmt(stats["min"]), _fmt(stats["max"])
        ),
        "",
        "Selected range: {} \u2013 {}".format(
            _fmt(selected_minimum),
            _fmt(selected_maximum),
        ),
        "Inside range: {:,} voxels ({:.2f}%)\n"
        "Below range: {:,} voxels ({:.2f}%)    "
        "Above range: {:,} voxels ({:.2f}%)    "
        "Total valid voxels: {:,}".format(
            stats["inside_count"], stats["inside_fraction"] * 100.0,
            stats["below_count"], stats["below_fraction"] * 100.0,
            stats["above_count"], stats["above_fraction"] * 100.0,
            stats["total"],
        ),
    ]
    if mixture is not None:
        summary_lines.append(
            "Populations (Gaussian mixture, {} component(s), {}):".format(
                mixture["n_components"],
                "BIC-selected" if n_components is None else "forced",
            )
        )
        for index, component in enumerate(mixture["components"]):
            summary_lines.append(
                "  P{} ({:.1f}%): {} \u00b1 {}{}".format(
                    index + 1,
                    component["weight"] * 100.0,
                    _format_value(component["mean"]),
                    _format_value(component["std"]),
                    " " + unit_suffix if unit_suffix else "",
                )
            )
    if exclusion_note:
        summary_lines.append(exclusion_note)
    ax_text.text(
        0.01, 0.95,
        "\n".join(summary_lines),
        transform=ax_text.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "black",
        },
    )

    return fig, stats


def _kmeans_2d(points, n_clusters, iterations=100, seed=0):
    """
    Minimal NumPy-only k-means for 2D points (no scikit-learn dependency,
    which is not guaranteed to be available inside Dragonfly's embedded
    Python). Returns (centroids, labels).
    """
    rng = np.random.default_rng(seed)
    n_points = points.shape[0]
    n_clusters = max(1, min(int(n_clusters), n_points))

    start_indices = rng.choice(n_points, size=n_clusters, replace=False)
    centroids = points[start_indices].copy()

    labels = np.zeros(n_points, dtype=np.int64)
    for _ in range(iterations):
        distances = np.linalg.norm(
            points[:, None, :] - centroids[None, :, :], axis=2
        )
        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels

        for cluster_index in range(n_clusters):
            members = points[labels == cluster_index]
            if members.size:
                centroids[cluster_index] = members.mean(axis=0)

    return centroids, labels


def create_crossplot_figure(
    zeff_values,
    rhoe_values,
    zeff_selected_minimum,
    zeff_selected_maximum,
    rhoe_selected_minimum,
    rhoe_selected_maximum,
    n_clusters=None,
    rhoe_unit_suffix="",
    exclusion_note="",
):
    """
    Z_eff vs rho_e crossplot restricted to voxels inside BOTH selected
    ranges: density hexbin (log colour scale, so minor populations stay
    visible), marginal histograms on both axes, and the populations of a
    2D Gaussian mixture (BIC-selected unless n_clusters is given) drawn
    as 1-sigma / 2-sigma ellipses with their centres numbered.

    rho_e must be passed in the units named by rhoe_unit_suffix (e.g.
    values in 10^23 e-/cm3 with suffix "x10^23 e-.cm-3").
    Returns (figure, population_summary).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Ellipse

    zeff_values = np.asarray(zeff_values, dtype=np.float64).ravel()
    rhoe_values = np.asarray(rhoe_values, dtype=np.float64).ravel()

    valid = (
        np.isfinite(zeff_values)
        & np.isfinite(rhoe_values)
        & (zeff_values >= zeff_selected_minimum)
        & (zeff_values <= zeff_selected_maximum)
        & (rhoe_values >= rhoe_selected_minimum)
        & (rhoe_values <= rhoe_selected_maximum)
    )
    zeff_inside = zeff_values[valid]
    rhoe_inside = rhoe_values[valid]

    if zeff_inside.size == 0:
        raise ValueError(
            "No voxels fall inside both selected ranges -- nothing to plot."
        )

    rhoe_unit = " " + rhoe_unit_suffix if rhoe_unit_suffix else ""
    rhoe_axis_label = r"$\rho_e$" + (
        " ({})".format(rhoe_unit_suffix) if rhoe_unit_suffix else ""
    )

    mixture = fit_gaussian_mixture_2d(
        np.column_stack((zeff_inside, rhoe_inside)),
        n_components=n_clusters,
        max_components=3,
    )

    fig = plt.figure(figsize=(12, 11))
    grid = fig.add_gridspec(
        3, 2, width_ratios=(4, 1), height_ratios=(1, 4, 1.3),
        hspace=0.08, wspace=0.08,
    )
    ax = fig.add_subplot(grid[1, 0])
    ax_top = fig.add_subplot(grid[0, 0], sharex=ax)
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax)
    ax_text = fig.add_subplot(grid[2, :])
    ax_text.axis("off")

    hexbin = ax.hexbin(
        zeff_inside, rhoe_inside, gridsize=90, cmap="viridis",
        mincnt=1, norm=LogNorm(),
    )
    colorbar_axis = ax_top.inset_axes([1.03, 0.0, 0.03, 1.0])
    fig.colorbar(hexbin, cax=colorbar_axis, label="Voxel count (log)")

    population_colors = plt.cm.tab10(np.arange(len(mixture["components"])))
    zeff_grid = np.linspace(zeff_inside.min(), zeff_inside.max(), 400)
    rhoe_grid = np.linspace(rhoe_inside.min(), rhoe_inside.max(), 400)

    ax_top.hist(zeff_inside, bins=150, density=True, color="lightgray")
    ax_right.hist(
        rhoe_inside, bins=150, density=True, color="lightgray",
        orientation="horizontal",
    )

    summary = []
    for index, component in enumerate(mixture["components"]):
        colour = population_colors[index]
        mean = component["mean"]
        cov = component["cov"]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        for n_sigma, style in ((1, "-"), (2, "--")):
            width, height = 2 * n_sigma * np.sqrt(np.maximum(eigenvalues[::-1], 0))
            ax.add_patch(Ellipse(
                mean, width, height, angle=angle, fill=False,
                edgecolor=colour, linewidth=2, linestyle=style,
            ))
        ax.scatter(
            [mean[0]], [mean[1]], marker="X", s=160, color=colour,
            edgecolor="white", linewidth=1.2, zorder=5,
        )
        ax.annotate(
            str(index + 1), xy=(mean[0], mean[1]), xytext=(7, 7),
            textcoords="offset points", fontsize=12, fontweight="bold",
            color="white",
        )
        zeff_sd = float(np.sqrt(cov[0, 0]))
        rhoe_sd = float(np.sqrt(cov[1, 1]))
        ax_top.plot(
            zeff_grid,
            component["weight"] / (zeff_sd * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((zeff_grid - mean[0]) / zeff_sd) ** 2),
            color=colour, linewidth=2,
        )
        ax_right.plot(
            component["weight"] / (rhoe_sd * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((rhoe_grid - mean[1]) / rhoe_sd) ** 2),
            rhoe_grid, color=colour, linewidth=2,
        )
        summary.append((
            index + 1, component["weight"], float(mean[0]), zeff_sd,
            float(mean[1]), rhoe_sd,
        ))

    ax.set_xlabel(r"$Z_{eff}$")
    ax.set_ylabel(rhoe_axis_label)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_yticks([])
    ax_right.tick_params(labelleft=False)
    ax_right.set_xticks([])
    ax_top.set_title(
        r"$Z_{eff}$ vs $\rho_e$ crossplot" + " \u2014 populations from a 2D "
        "Gaussian mixture (ellipses: 1\u03c3 solid, 2\u03c3 dashed)"
    )

    lines = [
        "Selected range: Z_eff {} \u2013 {};  rho_e {} \u2013 {}{}".format(
            _format_value(zeff_selected_minimum),
            _format_value(zeff_selected_maximum),
            _format_value(rhoe_selected_minimum),
            _format_value(rhoe_selected_maximum),
            rhoe_unit,
        ),
        "Populations: {} ({}; fit on {:,} of {:,} voxels)".format(
            mixture["n_components"],
            "BIC-selected" if n_clusters is None else "forced",
            mixture["n_fit"], zeff_inside.size,
        ),
    ]
    for number, weight, zeff_mean, zeff_sd, rhoe_mean, rhoe_sd in summary:
        lines.append(
            "  P{}  ({:.1f}%):  Z_eff = {} \u00b1 {}    "
            "rho_e = {} \u00b1 {}{}".format(
                number, weight * 100.0,
                _format_value(zeff_mean), _format_value(zeff_sd),
                _format_value(rhoe_mean), _format_value(rhoe_sd),
                rhoe_unit,
            )
        )
    if exclusion_note:
        lines.append(exclusion_note)
    ax_text.text(
        0.0, 0.95, "\n".join(lines), transform=ax_text.transAxes,
        fontsize=10, family="monospace", verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "black"},
    )

    return fig, summary
