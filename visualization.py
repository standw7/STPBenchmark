import numpy as np
import torch
from typing import Dict
import pandas as pd
import matplotlib.patheffects as pe

import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["font.family"] = "monospace"
plt.rcParams.update({"font.size": 10})


def plot_with_contrast(ax, x, y, color="maroon", thick_lw=3.2, thin_lw=2.0, label=None):
    """Creates a line plot with thick/thin contrast effect."""
    ax.plot(x, y, color="black", lw=thick_lw, solid_capstyle="butt")
    ax.plot(x, y, color=color, lw=thin_lw, solid_capstyle="butt", label=label)


def fill_between_with_contrast(
    ax, x, y1, y2, color="maroon", alpha=0.5, label=None, edge_lw=0.5
):
    """Creates a fill-between with both solid fill and outlined edges."""
    ax.fill_between(x, y1, y2, color=color, alpha=alpha, label=label, lw=0)
    ax.plot(x, y1, color=color, lw=edge_lw, solid_capstyle="butt")
    ax.plot(x, y2, color=color, lw=edge_lw, solid_capstyle="butt")


def plot_optimization_trace(
    ax,
    traces: pd.DataFrame,
    color: str = "maroon",
    maximize: bool = True,
    label: str = None,
    full_legend: bool = True,
    legend_kwargs: dict = None,
) -> None:
    """
    Plot optimization results across multiple runs showing median and uncertainty bands.

    Args:
        ax: Matplotlib axis to plot on
        traces: DataFrame where each column represents an optimization trace
        color: Color to use for the plot elements
        maximize: If True, show best maximum found. If False, show best minimum found
        show_legend: Whether to show the legend
        legend_kwargs: Additional keyword arguments for legend customization
    """
    # Default legend settings
    legend_kwargs = legend_kwargs or {
        "framealpha": 1.0,
        "edgecolor": "black",
        "facecolor": "white",
        "loc": "best",
    }

    # Setup trial numbers for x-axis
    trials = np.arange(1, len(traces) + 1)

    # Calculate cumulative best values
    accumulator = np.maximum if maximize else np.minimum
    trajectories = accumulator.accumulate(traces.values, axis=0)

    # Calculate all percentiles at once for efficiency
    percentiles = np.percentile(trajectories, [5, 25, 50, 75, 95], axis=1)
    p5, p25, median, p75, p95 = percentiles

    # Plot median line with contrast effect
    plot_with_contrast(ax, trials, median, color=color)

    # Plot uncertainty bands
    fill_between_with_contrast(ax, trials, p25, p75, color=color, alpha=0.35)
    fill_between_with_contrast(ax, trials, p5, p95, color=color, alpha=0.35)

    # Add legend if requested
    if label is not None:
        # Create legend entries using plot elements
        plot_with_contrast(ax, [], [], color=color, label=f"{label} Median")
        if full_legend:
            ax.fill_between([], [], [], color=color, alpha=0.65, label="50% Interval")
            ax.fill_between([], [], [], color=color, alpha=0.35, label="90% Interval")
        ax.legend(**legend_kwargs)


def plot_top_values_discovery(
    ax,
    traces: pd.DataFrame,
    y: np.ndarray,
    top_percent: float = 5.0,
    color: str = "maroon",
    maximize: bool = True,
    label: str = None,
    full_legend: bool = True,
    legend_kwargs: dict = None,
) -> None:
    """
    Plot the percentage of top values discovered over optimization iterations.

    Args:
        ax: Matplotlib axis to plot on
        traces: DataFrame where each column represents an optimization trace
        y: Original target values to determine threshold
        top_percent: Percentage threshold for considering top values (0-100)
        color: Color to use for the plot elements
        maximize: If True, look for highest values. If False, look for lowest values
        show_legend: Whether to show the legend
        legend_kwargs: Additional keyword arguments for legend customization
    """
    if not 0 < top_percent < 100:
        raise ValueError("top_percent must be between 0 and 100")

    legend_kwargs = legend_kwargs or {
        "framealpha": 1.0,
        "edgecolor": "black",
        "facecolor": "white",
        "loc": "best",
    }

    trials = np.arange(1, len(traces) + 1)

    # Calculate threshold and total number of top values
    k = int(len(y) * (top_percent / 100))
    if maximize:
        # For maximization: get kth largest value
        threshold = np.partition(y, -k)[-k]
    else:
        # For minimization: get kth smallest value
        threshold = np.partition(y, k - 1)[k - 1]

    # Count total values meeting threshold
    n_top_total = np.sum(y >= threshold if maximize else y <= threshold)

    def get_discovery_trajectory(values):
        discovered = set()  # Track unique discoveries
        cumulative = []

        for val in values:
            if (maximize and val >= threshold) or (not maximize and val <= threshold):
                discovered.add(val)
            cumulative.append(len(discovered))

        return (np.array(cumulative) / n_top_total) * 100

    # Calculate trajectories for each run
    trajectories = np.array(
        [get_discovery_trajectory(traces[col].values) for col in traces.columns]
    ).T

    # Calculate all percentiles at once for efficiency
    percentiles = np.percentile(trajectories, [5, 25, 50, 75, 95], axis=1)
    p5, p25, median, p75, p95 = percentiles

    # Plot median line with contrast effect
    plot_with_contrast(ax, trials, median, color=color)

    # Plot uncertainty bands
    fill_between_with_contrast(ax, trials, p25, p75, color=color, alpha=0.35)
    fill_between_with_contrast(ax, trials, p5, p95, color=color, alpha=0.35)

    # Add legend if requested
    if label is not None:
        plot_with_contrast(ax, [], [], color=color, label=f"{label} Median")
        if full_legend:
            ax.fill_between([], [], [], color=color, alpha=0.65, label="50% Interval")
            ax.fill_between([], [], [], color=color, alpha=0.35, label="90% Interval")
        ax.legend(**legend_kwargs)
