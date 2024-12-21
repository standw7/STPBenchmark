import torch
import numpy as np
import pandas as pd
from models import VarSTP, VarGP
import matplotlib.pyplot as plt
from utils import TorchNormalizer
from runners import run_many_loops
from visualization import plot_optimization_trace, plot_top_values_discovery

# Load and preprocess data
data = np.loadtxt("data/P3HT_dataset.csv", delimiter=",", skiprows=1)

X = torch.tensor(data[:, 0:-1], dtype=torch.double)
y = torch.tensor(data[:, -1], dtype=torch.double).flatten()

normalizer = TorchNormalizer()
X = normalizer.fit_transform(X)

seed_list = np.loadtxt("random_seeds.txt", dtype=int)

results = run_many_loops(
    X,
    y,
    seeds=seed_list[:5],
    n_initial=10,
    n_trials=15,
    epochs=100,
    learning_rate=0.1,
    model_class=VarSTP,
)

# Create a figure to show optimization traces
fig, ax = plt.subplots(1, 2, figsize=(6, 4), dpi=200, width_ratios=[8, 1], sharey=True)
plot_optimization_trace(ax[0], results, torch.max(y), n_initial=10)
ax[1].scatter([0] * len(y), y, color="black", marker="o", s=10, alpha=0.5)
ax[1].set_xticks([])
plt.tight_layout()
plt.savefig("figures/STP_Trace.png", dpi=500)
plt.show()

fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
# Plot discovery rate
plot_top_values_discovery(ax, results, y, top_percent=5.0, n_initial=10)
plt.tight_layout()
plt.savefig("figures/STP_TopFound.png", dpi=500)
plt.show()


df = pd.DataFrame({key: value["y_selected"] for key, value in results.items()})
df.to_csv("results/STP2048Samples.csv", index=False)
