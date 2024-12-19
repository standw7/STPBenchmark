import numpy as np
from typing import Dict

import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["font.family"] = "monospace"
plt.rcParams.update({"font.size": 10})


def plot_with_contrast(ax, x, y, color="maroon", thick_lw=3.2, thin_lw=2.0, label=None):
    """Creates a line plot with thick/thin contrast effect."""
    ax.plot(x, y, color="black", lw=thick_lw)
    ax.plot(x, y, color=color, lw=thin_lw, label=label)


def fill_between_with_contrast(
    ax, x, y1, y2, color="maroon", alpha=0.5, label=None, edge_lw=1.0
):
    """Creates a fill-between with both solid fill and outlined edges."""
    ax.fill_between(x, y1, y2, color=color, alpha=alpha, label=label)
    ax.fill_between(x, y1, y2, fc="None", ec=color, lw=edge_lw)


def plot_optimization_results(
    ax,  # Now we pass in the axis instead of creating it
    all_results: Dict[int, dict],
    max_y: float,
    n_initial: int = 10,
) -> None:  # Return type changes since we're not returning fig, ax
    """
    Plot optimization results across multiple runs showing mean and uncertainty bands.

    Args:
        ax: Matplotlib axis to plot on
        all_results: Dictionary mapping seeds to their results dictionaries
        max_y: True maximum value in dataset for reference line
        n_initial: Number of initial samples for vertical line
    """
    # Remove figure creation since we're using passed axis

    # Rest of the function stays the same, just use the passed ax
    n_trials = len(next(iter(all_results.values()))["y_selected"])
    trials = np.arange(1, n_trials + 1)

    all_trajectories = []
    for results in all_results.values():
        trajectory = np.maximum.accumulate(results["y_selected"])
        all_trajectories.append(trajectory)

    all_trajectories = np.array(all_trajectories)
    mean_trajectory = np.mean(all_trajectories, axis=0)
    p25_trajectory = np.percentile(all_trajectories, 25, axis=0)
    p75_trajectory = np.percentile(all_trajectories, 75, axis=0)
    min_trajectory = np.min(all_trajectories, axis=0)
    max_trajectory = np.max(all_trajectories, axis=0)

    plot_with_contrast(ax, trials, mean_trajectory, label="Mean")
    fill_between_with_contrast(
        ax,
        trials,
        p25_trajectory,
        p75_trajectory,
        alpha=0.35,
        label="50%",
    )
    fill_between_with_contrast(
        ax,
        trials,
        min_trajectory,
        max_trajectory,
        alpha=0.35,
        label="100%",
    )

    ax.axhline(max_y, color="k", linestyle="--", label="True Maximum")
    ax.axvline(n_initial, color="k", linestyle="-", label="Initial Samples")

    ax.set_xlabel("Number of Trials")
    ax.set_ylabel("Best Value Found")
    ax.legend(framealpha=1.0, edgecolor="black", facecolor="white")
