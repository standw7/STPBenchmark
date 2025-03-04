# Benchmarking script for testing a single model on a single dataset - for debugging

import torch
import numpy as np
import pandas as pd
from models import VarTGP, VarGP, ExactGP
import matplotlib.pyplot as plt
from utils import TorchNormalizer
from runners import run_many_loops, run_single_loop
from visualization import plot_optimization_trace, plot_top_values_discovery


def run_benchmark(
    dataset_path: str,
    model_class,
    run_name,
    seed_list,
    n_initial=10,
    n_trials=20,
    epochs=200,
    learning_rate=0.05,
    invert_target=False,
):
    """
    Run benchmark experiment with improved data handling
    """
    # Load and preprocess data
    data = np.loadtxt(dataset_path, delimiter=",", skiprows=1)

    # average duplicate feature entries
    features, target = data[:, :-1], data[:, -1]
    unique_f, inv = np.unique(features, axis=0, return_inverse=True)
    data = np.column_stack(
        (unique_f, np.bincount(inv, weights=target) / np.bincount(inv))
    )

    X = torch.tensor(data[:, 0:-1], dtype=torch.double)
    y = torch.tensor(data[:, -1], dtype=torch.double).flatten()

    if invert_target:
        print(f"\nInverting target values for maximization problem")
        y = 1.0 / y

    X = TorchNormalizer().fit_transform(X)

    # Run optimization
    results = run_single_loop(
        X,
        y,
        seeds=seed_list,
        n_initial=n_initial,
        n_trials=n_trials,
        epochs=epochs,
        learning_rate=learning_rate,
        model_class=model_class,
    )

    # Convert y to numpy for plotting
    y_np = y.numpy()

    # Extract y_values for all seeds
    traces_df = results.xs("y_value", axis=1, level=1)

    # Save results
    dataset_name = dataset_path.split("/")[-1].split(".")[0]
    model_name = model_class.__name__
    results.to_csv(f"results/{model_name}_{dataset_name}_{run_name}.csv")

    # Plotting

    # Optimization traces
    fig, ax = plt.subplots(1, 2, figsize=(6, 4), width_ratios=[8, 1], sharey=True)

    plot_optimization_trace(ax[0], traces_df, color="#003f5c")
    ax[0].axvline(n_initial, color="black", linestyle="--")

    ax[1].scatter([0] * len(y_np), y_np, color="black", marker="o", s=10, alpha=0.5)
    ax[1].set_xticks([])

    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Target Value")

    plt.tight_layout()
    plt.show(block=False)

    # Top values discovery
    fig, ax = plt.subplots(figsize=(6, 4))

    plot_top_values_discovery(ax, traces_df, y_np, top_percent=5.0, color="#003f5c")
    ax.axvline(n_initial, color="black", linestyle="--")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Percentage of Top Values Discovered")

    plt.tight_layout()
    plt.show()
    plt.close()

    return results, traces_df


if __name__ == "__main__":
    # Example usage
    dataset = "data/AutoAM_dataset.csv"
    seed_list = np.loadtxt("random_seeds.txt", dtype=int)

    results, traces = run_benchmark(
        dataset_path=dataset,
        model_class=VarTGP,
        run_name="testing_file",
        seed_list=seed_list[:2],
        n_initial=10,
        n_trials=40,
        epochs=100,
        learning_rate=0.05,
        invert_target=False,
    )
