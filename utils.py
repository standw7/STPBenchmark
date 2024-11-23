# add imports here
import torch


def train_variational_gp(model, train_x, train_y, n_iterations=100, lr=0.1):
    """
    Train a variational GP model (STP or VariationalGP).

    Args:
        model: Variational GP model instance
        train_x: Training inputs
        train_y: Training targets
        n_iterations: Number of training iterations
        lr: Learning rate
    """
    # Update objective function with full dataset size
    model.update_objective(train_y)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": model.likelihood.parameters()}],
        lr=lr,
    )

    # Training loop
    model.train()
    model.likelihood.train()

    for i in range(n_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -model.objective_function(output, train_y)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{n_iterations} - Loss: {loss.item():.4f}")
