import torch
from botorch.test_functions import Hartmann

from models import VarSTP, VarGP, ExactGP
from runners import train_variational_gp

# Define the Hartmann6 function
hartmann = Hartmann(dim=6, negate=False)

# Initialize data
train_x = torch.rand(5, 6)
train_y = hartmann(train_x).unsqueeze(-1)

# Bayesian optimization loop
for _ in range(10):
    # Fit GP model
    model = VarSTP(train_x, train_y)
    mll = model.mll(model.train_inputs, model.train_targets)
    model.fit()

    # Define acquisition function
    UCB = UpperConfidenceBound(model, beta=0.1)

    # Optimize acquisition function
    candidate, _ = optimize_acqf(
        UCB,
        bounds=torch.tensor([[0.0] * 6, [1.0] * 6]),
        q=1,
        num_restarts=10,
        raw_samples=100,
    )

    # Evaluate and add to training data
    new_y = hartmann(candidate).unsqueeze(-1)
    train_x = torch.cat([train_x, candidate], dim=0)
    train_y = torch.cat([train_y, new_y], dim=0)
