import torch
import numpy as np
import pandas as pd
from models import VarSTP, VarGP, ExactGP
import matplotlib.pyplot as plt
from utils import TorchNormalizer
from runners import run_many_loops, run_single_loop
from visualization import plot_optimization_trace, plot_top_values_discovery

# Load and preprocess data
dataset = "CrossedBarrel_dataset.csv"
data = np.loadtxt(f"data/{dataset}", delimiter=",", skiprows=1)

# average duplicate feature entries
features, target = data[:, :-1], data[:, -1]
unique_f, inv = np.unique(features, axis=0, return_inverse=True)
data = np.column_stack((unique_f, np.bincount(inv, weights=target) / np.bincount(inv)))

X = torch.tensor(data[:, 0:-1], dtype=torch.double)
y = torch.tensor(data[:, -1], dtype=torch.double).flatten()

if dataset in ["Perovskite_dataset.csv", "AgNP_dataset.csv"]:
    print(f"\nInverting {dataset[:-4]} target values for maximization problem")
    y = 1.0 / y

X = TorchNormalizer().fit_transform(X)

seed_list = np.loadtxt("random_seeds.txt", dtype=int)

results = run_many_loops(
    X,
    y,
    seeds=seed_list[:2],
    n_initial=10,
    n_trials=25,
    epochs=200,
    learning_rate=0.05,
    model_class=ExactGP,
)

# Create a figure to show optimization traces
fig, ax = plt.subplots(1, 2, figsize=(6, 4), dpi=200, width_ratios=[8, 1], sharey=True)
plot_optimization_trace(ax[0], results, torch.max(y), n_initial=10)
ax[1].scatter([0] * len(y), y, color="black", marker="o", s=10, alpha=0.5)
ax[1].set_xticks([])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
# Plot discovery rate
plot_top_values_discovery(ax, results, y, top_percent=5.0, n_initial=10)
plt.tight_layout()
plt.show()

df = pd.DataFrame({key: value["y_selected"] for key, value in results.items()})
df.to_csv("results/STP2048Samples.csv", index=False)
