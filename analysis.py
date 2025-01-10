import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
from visualization import (
    plot_optimization_trace,
    plot_top_values_discovery,
    dummy_legend,
)

dataset_name = "P3HT"
dataset = pd.read_csv(f"data/{dataset_name}_dataset.csv")

results = os.listdir("results")
results = [result for result in results if dataset_name in result]

# PLOT OPTIMIZATION TRACES

fig, ax = plt.subplots(1, 2, figsize=(6, 4), dpi=200, width_ratios=[8, 1], sharey=True)

ax[0].set_title(dataset_name, loc="left")

for i, result in enumerate(results):
    model_name = result.split("_")[0]
    result_data = pd.read_csv(f"results/{result}")

    ax[0].plot(
        [],
        [],
        label=model_name,
        color=colors[i],
        path_effects=[pe.withStroke(linewidth=3.2, foreground="black")],
    )

    plot_optimization_trace(
        ax[0],
        result_data,
        n_initial=10,
        color=colors[i],
        label=False,
    )

ax[1].scatter(
    [0] * len(dataset.iloc[:, -1]),
    dataset.iloc[:, -1],
    color="black",
    marker="o",
    s=10,
    alpha=0.5,
)
ax[1].set_xticks([])

dummy_legend(ax[0])  # gen a dummy legend for plot features

plt.tight_layout()
plt.show()

# PLOT DISCOVERY RATES

fig, ax = plt.subplots(1, 2, figsize=(6, 4), dpi=200, width_ratios=[8, 1], sharey=True)

ax[0].set_title(dataset_name, loc="left")

for i, result in enumerate(results):
    model_name = result.split("_")[0]
    result_data = pd.read_csv(f"results/{result}")

    ax[0].plot(
        [],
        [],
        label=model_name,
        color=colors[i],
        path_effects=[pe.withStroke(linewidth=3.2, foreground="black")],
    )

    plot_top_values_discovery(
        ax[0],
        result_data,
        y=dataset.iloc[:, -1],
        n_initial=10,
        color=colors[i],
        label=False,
    )
dummy_legend(ax[0])  # gen a dummy legend for plot features
plt.tight_layout()
plt.show()
