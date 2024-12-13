import torch
import numpy as np
from models import VarSTP, VarGP
import matplotlib.pyplot as plt
from utils import TorchNormalizer
from runners import run_many_loops
from visualization import plot_optimization_results

# Load and preprocess data
data = np.loadtxt("data/P3HT_dataset.csv", delimiter=",", skiprows=1)

X = torch.tensor(data[:, 0:-1], dtype=torch.double)
y = torch.tensor(data[:, -1], dtype=torch.double).flatten()

normalizer = TorchNormalizer()
X = normalizer.fit_transform(X)

results = run_many_loops(
    X,
    y,
    seeds=[1, 25, 323, 42, 5643],
    n_initial=10,
    n_trials=90,
    epochs=100,
    learning_rate=0.1,
    model_class=VarSTP,
)

# Create figure with subplots
fig, ax = plt.subplots(1, 2, figsize=(7, 4), dpi=200, width_ratios=[5, 1], sharey=True)

# Use the optimization results plot in the first subplot
plot_optimization_results(ax[0], results, torch.max(y), n_initial=10)

# Create your second plot
ax[1].scatter([0] * len(y), y, color="black", marker="o", s=10, alpha=0.5)
ax[1].set_xticks([])

plt.tight_layout()
plt.show()
