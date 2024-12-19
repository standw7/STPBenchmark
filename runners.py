import torch
import numpy as np
from botorch.acquisition import LogExpectedImprovement
from models import VarSTP, VarGP  # Assuming these are your model classes
from utils import set_seeds, get_initial_samples
from optim import train_variational_model
from tqdm import tqdm, trange
from typing import Type, Union, Optional, List, Dict
import warnings


def run_single_loop(
    X: torch.Tensor,
    y: torch.Tensor,
    model_class: Type[Union[VarSTP, VarGP]],
    model_kwargs: Optional[dict] = None,
    n_initial: int = 10,
    n_trials: int = 25,
    epochs: int = 100,
    learning_rate: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Performs Bayesian optimization loop on given dataset.

    Args:
        X: Feature matrix
        y: Target values
        model_class: Class of the model to use (e.g., VarSTP or VarGP)
        model_kwargs: Optional dictionary of kwargs to pass to model initialization
        n_initial: Number of initial points to sample
        n_trials: Number of optimization trials
        epochs: Number of training epochs per trial
        learning_rate: Learning rate for model training
        seed: Random seed for reproducibility

    Returns:
        dict containing:
        - best_x: Best input point found
        - best_y: Best target value found
        - X_selected: History of selected points
        - y_selected: History of selected target values
        - trained_model: Final trained model
    """
    set_seeds(seed)

    # Ensure inputs are double tensors
    X = X.double()
    y = y.double().flatten()

    # Default model kwargs if none provided
    if model_kwargs is None:
        model_kwargs = {}

    # Get initial training data
    sampled_indices = get_initial_samples(y, n_samples=n_initial, percentile=95)
    mask = torch.ones(len(X), dtype=torch.bool)
    mask[sampled_indices] = False

    X_train = X[sampled_indices]
    y_train = y[sampled_indices]
    X_candidate = X[mask]
    y_candidate = y[mask]

    for _ in trange(n_trials):
        # Initialize and train model
        if "inducing_points" in model_class.__init__.__code__.co_varnames:
            model_kwargs["inducing_points"] = X_train

        model = model_class(**model_kwargs).double()
        train_variational_model(
            model, X_train, y_train, epochs=epochs, lr=learning_rate
        )

        # Get next point using LogExpectedImprovement
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acq_values = LogExpectedImprovement(
                model, best_f=y_train.max(), maximize=True
            ).forward(X_candidate.unsqueeze(1))

        # if observation noise, take mean of outputs
        if acq_values.ndim > 1:
            acq_values = torch.mean(acq_values, dim=0)

        best_idx = torch.argmax(acq_values)

        # Update training and candidate sets
        X_train = torch.cat([X_train, X_candidate[best_idx].unsqueeze(0)])
        y_train = torch.cat([y_train, y_candidate[best_idx].unsqueeze(0)])

        # Update candidate sets
        mask = torch.ones(len(X_candidate), dtype=torch.bool)
        mask[best_idx] = False
        X_candidate = X_candidate[mask]
        y_candidate = y_candidate[mask]

    # Find best result
    best_idx = torch.argmax(y_train)
    best_x = X_train[best_idx]
    best_y = y_train[best_idx]

    return {
        "best_x": best_x,
        "best_y": best_y,
        "X_selected": X_train,
        "y_selected": y_train,
        "trained_model": model,
    }


def run_many_loops(
    X: torch.Tensor,
    y: torch.Tensor,
    model_class: Type[Union[VarSTP, VarGP]],
    seeds: List[int],
    **kwargs,  # To pass through to run_single_loop
) -> Dict[int, dict]:
    """
    Performs multiple Bayesian optimization loops with different random seeds.

    Args:
        X: Feature matrix
        y: Target values
        model_class: Class of the model to use (e.g., VarSTP or VarGP)
        seeds: List of random seeds to use
        **kwargs: Additional arguments to pass to run_single_loop

    Returns:
        Dictionary mapping each seed to its results dictionary containing:
        - best_x: Best input point found
        - best_y: Best target value found
        - X_selected: History of selected points
        - y_selected: History of selected target values
        - trained_model: Final trained model
    """
    # Initialize dictionary to store results for each seed
    all_results = {}

    # Run optimization with each seed
    for seed in tqdm(seeds, desc="Running optimization loops"):
        try:
            # Run single optimization loop with current seed
            results = run_single_loop(
                X=X, y=y, model_class=model_class, seed=seed, **kwargs
            )
            all_results[seed] = results
        except Exception as e:
            print(f"Error encountered for seed {seed}: {e}")
            print("Continung with next seed...")

        # Store results for this seed

    return all_results
