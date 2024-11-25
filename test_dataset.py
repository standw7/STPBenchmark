import torch
import numpy as np
from botorch.acquisition import LogExpectedImprovement
from models import VarSTP, VarGP
from utils import set_seeds, get_initial_samples
from optim import train_variational_model
import matplotlib.pyplot as plt
from tqdm import trange
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

# # Get initial training data
# num_initial = 10
# sampled = get_initial_samples(y, n_samples=num_initial, percentile=95)
# mask = torch.ones(len(X), dtype=torch.bool)
# mask[sampled] = False

# X_train, y_train = X[sampled], y[sampled]
# X_candidate, y_candidate = X[mask], y[mask]

# num_trials = 25
# for i in trange(num_trials):

#     # Initialize and train model
#     model = VarSTP(inducing_points=X_train).double()
#     train_variational_model(model, X_train, y_train, epochs=500, lr=0.1)

#     # Get next point using LogExpectedImprovement
#     model.eval()
#     acq_values = LogExpectedImprovement(
#         model, best_f=y_train.max(), maximize=True
#     ).forward(X_candidate.unsqueeze(1))

#     best_idx = torch.argmax(acq_values)

#     # Update training and candidate sets
#     X_train = torch.cat([X_train, X_candidate[best_idx].unsqueeze(0)])
#     y_train = torch.cat([y_train, y_candidate[best_idx].unsqueeze(0)])

#     mask = torch.ones(len(X_candidate), dtype=torch.bool)
#     mask[best_idx] = False
#     X_candidate = X_candidate[mask]
#     y_candidate = y_candidate[mask]


# plt.plot(
#     np.arange(1, num_initial + num_trials + 1),
#     np.maximum.accumulate(y_train.numpy()),
#     marker=".",
# )
# plt.plot(
#     np.arange(1, num_initial + num_trials + 1),
#     y_train,
#     marker=".",
#     ls="None",
#     color="grey",
# )

# plt.axhline(torch.max(y), color="k", linestyle="--")
# plt.axvline(num_initial, color="k", linestyle="--")
# plt.show()
