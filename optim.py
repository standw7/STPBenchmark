import torch
import gpytorch
from tqdm import trange
from utils import TorchStandardScaler
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


def train_natural_variational_model(
    model, X, y, epochs=500, lr=0.05, verbose=False, y_standardize=True
):
    """
    Trains a GPyTorch model using variational inference with stochastic optimization.
    Keeps track of the best model state during training.

    Args:
    model: GPyTorch model to be trained
    X: Input training data tensor
    y: Target training data tensor
    epochs (int, optional): Number of training epochs. Defaults to 100.
    lr (float, optional): Learning rate for Adam optimizer. Defaults to 0.01.
    verbose (bool, optional): If True, returns training loss history and best epoch. Defaults to False.
    y_standardize (bool, optional): Whether to standardize target values. Defaults to True.

    Returns:
    - If verbose=True: tuple (loss_history, best_epoch)
    - If verbose=False: None
    """

    if y_standardize:
        y = TorchStandardScaler().fit_transform(y)

    model.train()
    model.likelihood.train()

    variational_ngd_optimizer = gpytorch.optim.NGD(
        model.variational_parameters(), num_data=y.size(0), lr=lr
    )

    hyperparameter_optimizer = torch.optim.Adam(
        [
            {"params": model.hyperparameters()},
        ],
        lr=0.01,
    )

    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=y.size(0))

    loss_history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    for i in range(epochs):
        variational_ngd_optimizer.zero_grad()
        hyperparameter_optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        current_loss = loss.item()
        loss_history.append(current_loss)

        # Update best model if we found a better loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {
                "model": model.state_dict(),
                "var_opt": variational_ngd_optimizer.state_dict(),
                "hyper_opt": hyperparameter_optimizer.state_dict(),
            }
            best_epoch = i

        loss.backward()
        variational_ngd_optimizer.step()
        hyperparameter_optimizer.step()

    # Load the best model state
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        variational_ngd_optimizer.load_state_dict(best_state["var_opt"])
        hyperparameter_optimizer.load_state_dict(best_state["hyper_opt"])

    if verbose:
        print(best_epoch)
        # return loss_history, best_epoch
    return None


class AnimatedLossPlot:
    def __init__(self, max_epochs):
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.max_epochs = max_epochs
        self.loss_history = []

        # Initialize the line
        (self.line,) = self.ax.plot([], [], "b-", label="Training Loss")

        # Set up the plot
        self.ax.set_xlim(0, max_epochs)
        self.ax.set_ylim(0, 1)  # You might want to adjust this based on your loss range
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Loss Over Time")
        self.ax.grid(True)
        self.ax.legend()

    def update_plot(self, loss_value):
        """Update the loss history and redraw the plot"""
        self.loss_history.append(loss_value)

        # Update line data
        x_data = list(range(len(self.loss_history)))
        self.line.set_data(x_data, self.loss_history)

        # Adjust y-axis limits if necessary
        if len(self.loss_history) > 1:
            min_loss = min(self.loss_history)
            max_loss = max(self.loss_history)
            margin = (max_loss - min_loss) * 0.1
            # self.ax.set_ylim(min_loss - margin, max_loss + margin)
            self.ax.set_ylim(0, 5)

        # Draw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
