import torch
import gpytorch
from tqdm import trange
from utils import TorchStandardScaler
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


def train_exact_model_botorch(model, X, y, y_standardize=True):
    """
    Trains an Exact GPyTorch model using exact marginal likelihood optimization as using the
    optimizer preferred within Botorch.

    Args:
    model: BoTorch model to be trained
    X: Input training data tensor
    y: Target training data tensor
    y_standardize (bool, optional): Whether to standardize target values. Defaults to True.

    Returns:
    None - Model is trained in-place
    """

    if y_standardize:
        y = TorchStandardScaler().fit_transform(y)

    model.train()
    model.likelihood.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)


def train_exact_model(
    model, X, y, epochs=100, lr=0.01, verbose=False, y_standardize=True
):

    if y_standardize:
        y = TorchStandardScaler().fit_transform(y)

    model.train()
    model.likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    loss_history = []
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()

    if verbose:
        return loss_history


def train_variational_model(
    model, X, y, epochs=100, lr=0.01, verbose=False, y_standardize=True
):
    """
    Trains a GPyTorch model using variational inference with stochastic optimization.

    Args:
    model: GPyTorch model to be trained
    X: Input training data tensor
    y: Target training data tensor
    epochs (int, optional): Number of training epochs. Defaults to 100.
    lr (float, optional): Learning rate for Adam optimizer. Defaults to 0.01.
    verbose (bool, optional): If True, returns training loss history. Defaults to False.
    y_standardize (bool, optional): Whether to standardize target values. Defaults to True.

    Returns:
    list, optional: Training loss history if verbose=True, otherwise None
    """

    if y_standardize:
        y = TorchStandardScaler().fit_transform(y)

    model.train()
    model.likelihood.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=y.size(0))

    loss_history = []
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()

    if verbose:
        return loss_history
