import torch
import gpytorch
from tqdm import trange
from utils import TorchStandardScaler
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


def train_exact_model_botorch(
    model, X, y, epochs=100, lr=0.01, verbose=False, y_standardize=True
):

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
