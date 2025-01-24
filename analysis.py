import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from visualization import plot_optimization_trace, plot_top_values_discovery

colors = ["#003f5c", "#bc5090", "#ffa600"]

dataset_name = "AutoAM"
dataset = pd.read_csv(f"data/{dataset_name}_dataset.csv")

# take the average target value of duplicate feature entries
features, target = dataset.iloc[:, :-1], dataset.iloc[:, -1]
unique_f, inv = np.unique(features, axis=0, return_inverse=True)
dataset = pd.DataFrame(
    np.column_stack((unique_f, np.bincount(inv, weights=target) / np.bincount(inv)))
)

results = os.listdir("results")
results = [result for result in results if dataset_name in result]

# PLOT OPTIMIZATION TRACES

fig, ax = plt.subplots(1, 2, figsize=(6, 4), dpi=200, width_ratios=[8, 1], sharey=True)

ax[0].set_title(dataset_name, loc="left")

for i, result in enumerate(results):

    model_name = result.split("_")[0]
    result_data = pd.read_csv(f"results/{result}", header=[0, 1], index_col=0)

    traces_df = result_data.xs("y_value", axis=1, level=1)

    plot_optimization_trace(
        ax=ax[0],
        traces=traces_df,
        color=colors[i],
        label=model_name,
        full_legend=True if i == 0 else False,
    )

ax[0].axvline(10, color="k", ls="--")

ax[1].scatter(
    [0] * len(dataset.iloc[:, -1]),
    dataset.iloc[:, -1],
    color="black",
    marker="o",
    s=10,
    alpha=0.5,
)
ax[1].set_xticks([])

plt.tight_layout()
plt.show()

# PLOT DISCOVERY RATES

fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

ax.set_title(dataset_name, loc="left")

for i, result in enumerate(results):
    model_name = result.split("_")[0]
    result_data = pd.read_csv(f"results/{result}", header=[0, 1], index_col=0)

    traces_df = result_data.xs("y_value", axis=1, level=1)

    plot_top_values_discovery(
        ax=ax,
        traces=traces_df,
        y=dataset.iloc[:, -1],
        color=colors[i],
        label=model_name,
        full_legend=True if i == 0 else False,
    )

plt.tight_layout()
plt.show()
