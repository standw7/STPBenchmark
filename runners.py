import torch
import numpy as np
import pandas as pd
from typing import Type, Union, Optional, List, Dict, Tuple
from tqdm import tqdm, trange
import warnings
from botorch.acquisition import LogExpectedImprovement
from models import VarTGP, VarGP, ExactGP
from utils import set_seeds, get_initial_samples
from optim import train_exact_model_botorch, train_natural_variational_model


def run_single_loop(
    X: torch.Tensor,
    y: torch.Tensor,
    model_class: Type[Union[VarTGP, VarGP, ExactGP]],
    model_kwargs: Optional[dict] = None,
    n_initial: int = 10,
    n_trials: int = 25,
    epochs: int = 100,
    learning_rate: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[float]]:
    """
    Performs Bayesian optimization loop on given dataset with improved error handling.

    Returns:
        Tuple containing:
        - selected_indices: List of indices selected during optimization
        - selected_values: List of corresponding y values
    """
    set_seeds(seed)
    X = X.double()
    y = y.double().flatten()

    if model_kwargs is None:
        model_kwargs = {}

    # Initialize results tracking
    selected_indices = []
    selected_values = []

    try:
        # Get initial training data
        initial_indices = get_initial_samples(y, n_samples=n_initial, percentile=50)
        selected_indices.extend(initial_indices.tolist())
        selected_values.extend(y[initial_indices].tolist())

        mask = torch.ones(len(X), dtype=torch.bool)
        mask[initial_indices] = False

        X_train = X[selected_indices]
        y_train = y[selected_indices]
        X_candidate = X[mask]
        y_candidate = y[mask]

        # Adjust n_trials if necessary
        n_trials = min(n_trials, len(X) - n_initial)

        for trial in range(n_trials):
            # Set up model
            if "inducing_points" in model_class.__init__.__code__.co_varnames:
                model_kwargs["inducing_points"] = X_train

            if model_class == ExactGP:
                model_kwargs["X_train"] = X_train
                model_kwargs["y_train"] = y_train
                model_kwargs["input_transform"] = None

            model = model_class(**model_kwargs).double()

            # Train model
            if model_class == ExactGP:
                train_exact_model_botorch(model, X_train, y_train)
            else:
                train_natural_variational_model(
                    model, X_train, y_train, epochs=epochs, lr=learning_rate
                )

            # Get next point
            model.eval()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                acq_values = LogExpectedImprovement(
                    model, best_f=y_train.max(), maximize=True
                ).forward(X_candidate.unsqueeze(1))

            if hasattr(model, "num_likelihood_samples"):
                acq_values = torch.mean(acq_values, dim=0)

            best_idx = torch.argmax(acq_values)

            # Get original index from remaining candidates
            original_idx = torch.where(mask)[0][best_idx]
            selected_indices.append(original_idx.item())
            selected_values.append(y[original_idx].item())

            # Update training data
            X_train = torch.cat([X_train, X_candidate[best_idx].unsqueeze(0)])
            y_train = torch.cat([y_train, y_candidate[best_idx].unsqueeze(0)])

            # Update candidate sets
            mask[original_idx] = False
            X_candidate = X[mask]
            y_candidate = y[mask]

        return selected_indices, selected_values

    except Exception as e:
        print(f"Error occurred at iteration {len(selected_values)}: {str(e)}")
        return selected_indices, selected_values


def run_many_loops(
    X: torch.Tensor,
    y: torch.Tensor,
    model_class: Type[Union[VarTGP, VarGP, ExactGP]],
    seeds: List[int],
    **kwargs,
) -> pd.DataFrame:
    """
    Runs multiple optimization loops and returns results in a DataFrame with seed as column index.

    Returns:
        DataFrame with iteration as index and seeds as columns, containing two levels:
        - x_index: Index of selected point in original dataset
        - y_value: Corresponding y value
    """
    n_initial = kwargs.get("n_initial", 10)
    n_trials = kwargs.get("n_trials", 25)
    total_iterations = n_initial + n_trials

    # Initialize data storage
    data_dict = {
        key: []
        for seed in seeds
        for key in [
            ("seed_{}".format(seed), "x_index"),
            ("seed_{}".format(seed), "y_value"),
        ]
    }

    # Create index for iterations
    index = list(range(total_iterations))

    for seed in tqdm(seeds, desc="Running optimization loops"):
        indices, values = run_single_loop(
            X=X, y=y, model_class=model_class, seed=seed, **kwargs
        )

        # Pad with NaN if necessary
        indices.extend([np.nan] * (total_iterations - len(indices)))
        values.extend([np.nan] * (total_iterations - len(values)))

        # Store results
        data_dict[("seed_{}".format(seed), "x_index")] = indices
        data_dict[("seed_{}".format(seed), "y_value")] = values

    # Create DataFrame with multi-level columns
    df = pd.DataFrame(data_dict, index=index)
    df.index.name = "iteration"

    return df
