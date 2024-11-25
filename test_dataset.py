import torch
import numpy as np
from models import VarSTP, VarGP
import matplotlib.pyplot as plt

from runners import run_campaign

# Load and preprocess data
data = np.loadtxt("data/P3HT_dataset.csv", delimiter=",", skiprows=1)
X = torch.tensor(data[:, 0:-1], dtype=torch.double)
y = torch.tensor(data[:, -1], dtype=torch.double).flatten()

results = run_campaign(
    X,
    y,
    n_initial=10,
    n_trials=25,
    epochs=100,
    learning_rate=0.1,
    model_class=VarSTP,
    seed=42,
)

fig, ax = plt.subplots(dpi=200, figsize=(6, 4))
trials = np.arange(1, len(results["y_selected"]) + 1)
ax.plot(trials, results["y_selected"], marker="o", ls="None", color="k")
ax.plot(trials, np.maximum.accumulate(results["y_selected"]), marker="None", lw=3)
ax.axhline(torch.max(y), color="k", linestyle="--")
ax.axvline(10, color="k", linestyle="--")
plt.show()
