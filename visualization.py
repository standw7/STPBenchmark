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
    ax, x, y1, y2, color="maroon", alpha=0.5, label=None, edge_lw=1.0
):
    """Creates a fill-between with both solid fill and outlined edges."""
    ax.fill_between(x, y1, y2, color=color, alpha=alpha, label=label, lw=0)
    ax.plot(x, y1, color=color, lw=edge_lw, solid_capstyle="butt")
    ax.plot(x, y2, color=color, lw=edge_lw, solid_capstyle="butt")


def dummy_legend(ax):
    """Add a dummy legend to the plot."""
    ax.plot(
        [],
        [],
        color="grey",
        lw=2,
        label="Median",
        path_effects=[pe.withStroke(linewidth=3.2, foreground="black")],
    )
    ax.fill_between([], [], [], color="grey", alpha=0.35, label="50% Interval")
    ax.fill_between([], [], [], color="grey", alpha=0.35, label="95% Interval")
    return ax.legend(framealpha=1.0, edgecolor="black", facecolor="white")


def plot_optimization_trace(
    ax,
    results_df: pd.DataFrame,  # Changed from Dict to DataFrame
    n_initial: int = 10,
    color: str = "maroon",
    maximize: bool = True,
    label: bool = False,
) -> None:
    """
    Plot optimization results across multiple runs showing mean and uncertainty bands.

    Args:
        ax: Matplotlib axis to plot on
        results_df: DataFrame where each column represents a random seed's optimization trace
        max_y: True maximum value in dataset for reference line
        n_initial: Number of initial samples for vertical line
    """
    # Get number of trials from DataFrame length
    n_trials = len(results_df)
    trials = np.arange(1, n_trials + 1)

    # Convert DataFrame to numpy array for calculations
    # Each column becomes a trajectory
    if maximize:
        all_trajectories = np.maximum.accumulate(results_df.values, axis=0)
    else:
        all_trajectories = np.minimum.accumulate(results_df.values, axis=0)

    # Calculate statistics across seeds (axis=1 because now seeds are columns)
    mean_trajectory = np.mean(all_trajectories, axis=1)
    median_trajectory = np.median(all_trajectories, axis=1)
    p25_trajectory = np.percentile(all_trajectories, 25, axis=1)
    p75_trajectory = np.percentile(all_trajectories, 75, axis=1)
    p2_5_trajectory = np.percentile(all_trajectories, 2.5, axis=1)
    p97_5_trajectory = np.percentile(all_trajectories, 97.5, axis=1)

    plot_with_contrast(
        ax, trials, median_trajectory, label="Median" if label else None, color=color
    )
    fill_between_with_contrast(
        ax,
        trials,
        p25_trajectory,
        p75_trajectory,
        alpha=0.35,
        label="50% Interval" if label else None,
        color=color,
    )
    fill_between_with_contrast(
        ax,
        trials,
        p2_5_trajectory,
        p97_5_trajectory,
        alpha=0.35,
        label="95% Interval" if label else None,
        color=color,
    )

    ax.axvline(
        n_initial, color="k", linestyle="-", label="Initial Samples" if label else None
    )

    ax.set_xlabel("Number of Trials")
    ax.set_ylabel("Best Value Found")


def plot_top_values_discovery(
    ax,
    results_df: pd.DataFrame,
    y: torch.Tensor,
    top_percent: float = 5.0,
    n_initial: int = 10,
    color: str = "maroon",
    label: bool = False,
) -> None:
    """
    Plot the percentage of top values discovered over optimization iterations.

    Args:
        ax: Matplotlib axis to plot on
        results_df: DataFrame where each column represents a random seed's optimization trace
        y: Original target values tensor to determine threshold
        top_percent: Percentage threshold for considering top values (default: 5%)
        n_initial: Number of initial samples for vertical line
        color: Color to use for plotting (default: "maroon")
        label: Whether to add labels to the plot elements (default: False)
    """
    # Calculate threshold using PyTorch
    k = int(len(y) * (1 - top_percent / 100))
    threshold = np.partition(y, k)[k]
    n_top_total = np.sum(y >= threshold)

    # Get number of trials from DataFrame length
    n_trials = len(results_df)
    trials = np.arange(1, n_trials + 1)

    # Calculate discovery trajectories for each seed (column)
    all_trajectories = []
    for col in results_df.columns:
        selected_values = results_df[col].values
        cumulative_found = np.zeros(n_trials)

        unique_values = set()
        for i, val in enumerate(selected_values):
            if val >= threshold:
                unique_values.add(val)
            cumulative_found[i] = len(unique_values)

        # Convert to percentage
        trajectory = (cumulative_found / n_top_total) * 100
        all_trajectories.append(trajectory)

    # Convert to numpy array for calculations
    all_trajectories = np.array(
        all_trajectories
    ).T  # Transpose to match plot_optimization_trace

    # Calculate statistics across seeds
    median_trajectory = np.median(all_trajectories, axis=1)
    p25_trajectory = np.percentile(all_trajectories, 25, axis=1)
    p75_trajectory = np.percentile(all_trajectories, 75, axis=1)
    p2_5_trajectory = np.percentile(all_trajectories, 2.5, axis=1)
    p97_5_trajectory = np.percentile(all_trajectories, 97.5, axis=1)

    # Plot using existing style functions
    plot_with_contrast(
        ax,
        trials,
        median_trajectory,
        color=color,
        label=f"Median (top {top_percent}%)" if label else None,
    )
    fill_between_with_contrast(
        ax,
        trials,
        p25_trajectory,
        p75_trajectory,
        color=color,
        alpha=0.35,
        label="50% Interval" if label else None,
    )
    fill_between_with_contrast(
        ax,
        trials,
        p2_5_trajectory,
        p97_5_trajectory,
        color=color,
        alpha=0.35,
        label="95% Interval" if label else None,
    )

    # Add reference line for initial samples
    ax.axvline(
        n_initial, color="k", linestyle="-", label="Initial Samples" if label else None
    )

    # Customize plot
    ax.set_xlabel("Number of Trials")
    ax.set_ylabel(f"% of Top {top_percent}% Values Found")
    ax.set_ylim(0, 100)
