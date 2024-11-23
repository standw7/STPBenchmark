import torch
import gpytorch
from tqdm import trange


def train_variational_model(model, X, y, epochs=1000, lr=0.01, verbose=False):
    model.train(), model.likelihood.train()

    hyperparameter_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=y.size(0))

    loss_history = []
    for i in trange(epochs):
        ### Perform Adam step to optimize hyperparameters
        hyperparameter_optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss_history.append(loss.item())
        loss.backward()
        hyperparameter_optimizer.step()

    if verbose:
        return loss_history
