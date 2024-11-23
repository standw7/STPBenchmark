import torch
from botorch.test_functions import Hartmann
import gpytorch
import torch
import matplotlib.pyplot as plt
import gpytorch
from tqdm import trange

from models import STP
from utils import set_seeds
from optim import train_variational_model

set_seeds(0)  # fix the random seed

# Initialize Hartmann6 function
hartmann = Hartmann(dim=6, negate=True)

X_train = torch.rand(100, 6)
y_train = hartmann(X_train).flatten()

model = STP(inducing_points=X_train)

train_variational_model(model, X_train, y_train, epochs=1000, lr=0.1)

X_test = torch.rand(100, 6)
y_test = hartmann(X_test).flatten()

model.eval()
model.likelihood.eval()
with torch.no_grad():
    train_preds = model(X_train)
    test_preds = model(X_test)

print("Train MAE:", torch.nn.L1Loss()(y_train, train_preds.mean))
print("Test MAE:", torch.nn.L1Loss()(y_test, test_preds.mean))
