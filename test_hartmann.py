import torch
from botorch.test_functions import Hartmann
import gpytorch
import torch
import matplotlib.pyplot as plt
import gpytorch
from tqdm import trange

from models import VarSTP, VarGP
from utils import set_seeds
from optim import train_variational_model
# standardize the tensor
from utils import TorchStandardScaler
from utils import TorchNormalizer

set_seeds(0)  # fix the random seed

# Initialize Hartmann6 function
hartmann = Hartmann(dim=6, negate=True)

X_train = torch.rand(100, 6)
y_train = hartmann(X_train).flatten()

# Initialize and fit the normalizer on the training data
X_normalizer = TorchNormalizer()
y_normalizer = TorchNormalizer()

X_train_normalized = X_normalizer.fit_transform(X_train)
y_train_normalized = y_normalizer.fit_transform(y_train)

model = VarSTP(inducing_points=X_train_normalized)

train_variational_model(model, X_train_normalized, y_train_normalized, epochs=1000, lr=0.1)

X_test = torch.rand(100, 6)
y_test = hartmann(X_test).flatten()

# Use the same normalizer instance to transform the test data
X_test_normalized = X_normalizer.transform(X_test)
y_test_normalized = y_normalizer.transform(y_test)

model.eval()
model.likelihood.eval()
with torch.no_grad():
    train_preds = model(X_train_normalized)
    test_preds = model(X_test_normalized)

print("Train MAE:", torch.nn.L1Loss()(y_train_normalized, train_preds.mean))
print("Test MAE:", torch.nn.L1Loss()(y_test_normalized, test_preds.mean))

# without normalization: Train MAE: 0.1288, Test MAE: 0.1158
# with normalization: Train MAE: 0.0739, Test MAE: 0.0656
# with standardization: Train MAE: 0.3363, Test MAE: 0.3048 
# with normalization and standardization: Train MAE: 0.3376, Test MAE: 0.3054

