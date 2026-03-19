"""
Bias Detection Visualizations.

Generates matplotlib/seaborn charts for bias reports. All functions
return matplotlib Figure objects so callers can log them to MLflow
via tracker.log_figure(fig, filename).

Charts:
  - Per-slice metric comparison (grouped bar)
  - Disparity heatmap (metric × slice)
  - Per-group fairness breakdown (horizontal bar)
  - Issuer/card-type distribution (stacked bar)
  - Explanation quality comparison (box/violin)
  - Before/after mitigation comparison (paired bar)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for CI/server
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    MPL_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    MPL_AVAILABLE = False
    logger.warning("matplotlib not installed — bias visualizations disabled")

try:
    import seaborn as sns

    SNS_AVAILABLE = True
except ImportError:
    sns = None  # type: ignore[assignment]
    SNS_AVAILABLE = False
    logger.warning("seaborn not installed — using matplotlib only")

# Colorblind-friendly palette (Wong 2011, Nature Methods)
# Safe for deuteranopia, protanopia, and tritanopia
COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# Pass/fail colors (colorblind-safe: blue vs orange instead of green vs red)
COLOR_PASS = "#0072B2"  # blue — within threshold
COLOR_FAIL = "#D55E00"  # vermillion — exceeds threshold
COLOR_BEFORE = "#D55E00"  # vermillion — before mitigation
COLOR_AFTER = "#0072B2"  # blue — after mitigation


def _check_mpl() -> None:
    if not MPL_AVAILABLE:
        raise ImportError(
            "matplotlib required for visualizations: pip install matplotlib"
        )


# =====================================================================
# SliceEvaluator charts
# =====================================================================


def plot_slice_metrics(
    slices: List[Dict[str, Any]],
    metrics: Sequence[str] = ("ndcg_5", "precision_5", "recall_5"),
    overall: Optional[Dict[str, float]] = None,
    title: str = "Per-Slice Model Metrics",
    figsize: tuple = (12, 6),
) -> Any:
    """
    Grouped bar chart comparing metrics across data slices.

    Parameters
    ----------
    slices : list[dict]
        Each dict has "name" and "metrics" keys (from SliceEvaluationReport).
    metrics : sequence[str]
        Which metrics to plot.
    overall : dict, optional
        Overall metric values — drawn as horizontal dashed lines.
    title : str
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()

    slice_names = [s["name"] for s in slices]
    n_slices = len(slice_names)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_slices)
    width = 0.8 / n_metrics

    for i, metric in enumerate(metrics):
        values = [s["metrics"].get(metric, 0) for s in slices]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=metric,
            color=COLORS[i % len(COLORS)],
            alpha=0.85,
        )
        # Value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    # Overall baselines
    if overall:
        for i, metric in enumerate(metrics):
            if metric in overall:
                ax.axhline(
                    y=overall[metric],
                    color=COLORS[i % len(COLORS)],
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                )

    ax.set_xlabel("Data Slice")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(slice_names, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, min(1.15, ax.get_ylim()[1] * 1.15))
    fig.tight_layout()
    return fig


def plot_disparity_heatmap(
    slices: List[Dict[str, Any]],
    overall: Dict[str, float],
    metrics: Sequence[str] = ("ndcg_5", "precision_5", "recall_5"),
    title: str = "Metric Deviation from Overall (% Difference)",
    figsize: tuple = (10, 6),
) -> Any:
    """
    Heatmap showing how each slice deviates from the overall metric.

    Cell values are (slice_value - overall) / overall as percentages.
    Red = underperforming, blue = overperforming.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()

    slice_names = [s["name"] for s in slices]
    data = []
    for s in slices:
        row = []
        for m in metrics:
            s_val = s["metrics"].get(m, 0)
            o_val = overall.get(m, 0)
            if o_val != 0:
                row.append((s_val - o_val) / abs(o_val) * 100)
            else:
                row.append(0)
        data.append(row)

    data_arr = np.array(data)

    fig, ax = plt.subplots(figsize=figsize)

    if SNS_AVAILABLE:
        sns.heatmap(
            data_arr,
            annot=True,
            fmt=".1f",
            center=0,
            cmap="PuOr",
            xticklabels=list(metrics),
            yticklabels=slice_names,
            ax=ax,
            cbar_kws={"label": "% Deviation"},
        )
    else:
        im = ax.imshow(data_arr, cmap="PuOr", aspect="auto", vmin=-50, vmax=50)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(list(metrics))
        ax.set_yticks(range(len(slice_names)))
        ax.set_yticklabels(slice_names)
        for i in range(len(slice_names)):
            for j in range(len(metrics)):
                ax.text(
                    j, i, f"{data_arr[i, j]:.1f}%", ha="center", va="center", fontsize=8
                )
        fig.colorbar(im, ax=ax, label="% Deviation")

    ax.set_title(title)
    fig.tight_layout()
    return fig


# =====================================================================
# ModelBiasDetector charts
# =====================================================================


def plot_fairness_metrics(
    per_group_metrics: Dict[str, Dict[str, float]],
    title: str = "Per-Group Fairness Breakdown",
    figsize: tuple = (10, 5),
) -> Any:
    """
    Horizontal bar chart of per-group metrics from Fairlearn MetricFrame.

    Parameters
    ----------
    per_group_metrics : dict[str, dict[str, float]]
        Outer key: sensitive feature name.
        Inner key: group name → metric value.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()

    n_features = len(per_group_metrics)
    fig, axes = plt.subplots(1, max(n_features, 1), figsize=figsize, squeeze=False)

    for idx, (feat_name, group_vals) in enumerate(per_group_metrics.items()):
        ax = axes[0][idx]
        groups = list(group_vals.keys())
        values = list(group_vals.values())
        colors = [COLORS[i % len(COLORS)] for i in range(len(groups))]

        bars = ax.barh(groups, values, color=colors, alpha=0.85)

        for bar, val in zip(bars, values):
            ax.text(
                val + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=8,
            )

        ax.set_xlabel("Metric Value")
        ax.set_title(f"{feat_name}", fontsize=10)
        ax.set_xlim(0, max(values) * 1.2 if values else 1)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_bias_summary(
    metrics: List[Dict[str, Any]],
    title: str = "Bias Detection Summary",
    figsize: tuple = (10, 5),
) -> Any:
    """
    Bar chart showing all bias metric values vs their thresholds.

    Green = within threshold, red = exceeds threshold.

    Parameters
    ----------
    metrics : list[dict]
        Each dict has "name", "sensitive_feature", "value",
        "threshold", "is_biased".

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()

    if not metrics:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No bias metrics to display", ha="center", va="center")
        ax.set_axis_off()
        return fig

    labels = [f"{m['name']}\n({m['sensitive_feature']})" for m in metrics]
    values = [m["value"] for m in metrics]
    thresholds = [m["threshold"] for m in metrics]
    colors = ["#D55E00" if m["is_biased"] else "#0072B2" for m in metrics]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))

    ax.bar(x, values, color=colors, alpha=0.85, label="Measured Value")
    ax.scatter(
        x, thresholds, color="black", marker="_", s=200, zorder=5, label="Threshold"
    )

    for i, (val, thresh) in enumerate(zip(values, thresholds)):
        ax.text(i, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# =====================================================================
# Scoring engine & LLM charts
# =====================================================================


def plot_issuer_distribution(
    metrics: List[Dict[str, Any]],
    title: str = "Card Issuer Recommendation Distribution by Segment",
    figsize: tuple = (10, 6),
) -> Any:
    """
    Grouped bar chart showing issuer recommendation rates per segment.

    Parameters
    ----------
    metrics : list[dict]
        ComponentBiasMetric dicts with details containing
        "issuer" and "per_group_rates".

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()

    # Collect all groups and issuers
    all_groups: set = set()
    issuer_data: Dict[str, Dict[str, float]] = {}

    for m in metrics:
        issuer = m.get("details", {}).get("issuer", m.get("check", "unknown"))
        rates = m.get("details", {}).get("per_group_rates", {})
        if rates:
            issuer_data[issuer] = rates
            all_groups.update(rates.keys())

    if not issuer_data:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No issuer data to display", ha="center", va="center")
        ax.set_axis_off()
        return fig

    groups = sorted(all_groups)
    issuers = list(issuer_data.keys())
    n_issuers = len(issuers)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(groups))
    width = 0.8 / n_issuers

    for i, issuer in enumerate(issuers):
        values = [issuer_data[issuer].get(g, 0) for g in groups]
        offset = (i - n_issuers / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=issuer,
            color=COLORS[i % len(COLORS)],
            alpha=0.85,
        )

    ax.set_xlabel("User Segment")
    ax.set_ylabel("Recommendation Rate")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_explanation_quality(
    metrics: List[Dict[str, Any]],
    title: str = "LLM Explanation Quality by Segment",
    figsize: tuple = (12, 4),
) -> Any:
    """
    Multi-panel bar chart showing explanation quality metrics per segment.

    Parameters
    ----------
    metrics : list[dict]
        ComponentBiasMetric dicts with details containing
        per-group quality values.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()

    quality_metrics = [m for m in metrics if "details" in m]
    n_panels = len(quality_metrics)

    if n_panels == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No quality data to display", ha="center", va="center")
        ax.set_axis_off()
        return fig

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)

    for idx, m in enumerate(quality_metrics):
        ax = axes[0][idx]
        check_name = m.get("check", m.get("check_name", "metric"))

        # Find the per-group data in details
        details = m.get("details", {})
        per_group = None
        for key in details:
            if isinstance(details[key], dict) and key.startswith("per_group"):
                per_group = details[key]
                break

        if per_group is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        groups = list(per_group.keys())
        values = list(per_group.values())
        colors = [COLORS[i % len(COLORS)] for i in range(len(groups))]

        ax.bar(groups, values, color=colors, alpha=0.85)

        for i, val in enumerate(values):
            ax.text(i, val + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        # Overall line
        overall_key = [k for k in details if k.startswith("overall")]
        if overall_key:
            ax.axhline(
                y=details[overall_key[0]],
                color="black",
                linestyle="--",
                alpha=0.6,
                label="Overall",
            )
            ax.legend(fontsize=7)

        ax.set_title(check_name.replace("_", " ").title(), fontsize=9)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=7)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# =====================================================================
# Mitigation before/after charts
# =====================================================================


def plot_mitigation_comparison(
    before_metrics: Dict[str, float],
    after_metrics: Dict[str, float],
    title: str = "Bias Mitigation: Before vs After",
    figsize: tuple = (10, 5),
) -> Any:
    """
    Paired bar chart comparing metrics before and after mitigation.

    Parameters
    ----------
    before_metrics, after_metrics : dict[str, float]
        Metric name → value.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()

    all_keys = sorted(set(before_metrics) | set(after_metrics))
    before_vals = [before_metrics.get(k, 0) for k in all_keys]
    after_vals = [after_metrics.get(k, 0) for k in all_keys]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(all_keys))
    width = 0.35

    bars_before = ax.bar(
        x - width / 2,
        before_vals,
        width,
        label="Before",
        color=COLOR_BEFORE,
        alpha=0.75,
    )
    bars_after = ax.bar(
        x + width / 2,
        after_vals,
        width,
        label="After",
        color=COLOR_AFTER,
        alpha=0.75,
    )

    # Value labels
    for bar, val in zip(bars_before, before_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for bar, val in zip(bars_after, after_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_group_metric_comparison(
    before_groups: Dict[str, float],
    after_groups: Dict[str, float],
    metric_name: str = "accuracy",
    title: str = "Per-Group Metric: Before vs After Mitigation",
    figsize: tuple = (10, 5),
) -> Any:
    """
    Paired horizontal bar chart comparing per-group metrics before/after.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()

    all_groups = sorted(set(before_groups) | set(after_groups))
    before_vals = [before_groups.get(g, 0) for g in all_groups]
    after_vals = [after_groups.get(g, 0) for g in all_groups]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(all_groups))
    height = 0.35

    ax.barh(
        y - height / 2,
        before_vals,
        height,
        label="Before",
        color=COLOR_BEFORE,
        alpha=0.75,
    )
    ax.barh(
        y + height / 2,
        after_vals,
        height,
        label="After",
        color=COLOR_AFTER,
        alpha=0.75,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(all_groups)
    ax.set_xlabel(metric_name)
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
