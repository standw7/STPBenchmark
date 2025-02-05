import os
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
    output_path: str,
    invert_y: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Runs multiple optimization loops and saves results incrementally.

    Args:
        X: Input tensor
        y: Target tensor
        model_class: The model class to use
        seeds: List of random seeds
        output_path: Path to save the results file
        **kwargs: Additional arguments for run_single_loop

    Returns:
        DataFrame with iteration as index and seeds as columns
    """
    n_initial = kwargs.get("n_initial", 10)
    n_trials = kwargs.get("n_trials", 25)
    total_iterations = n_initial + n_trials

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing results if any
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, header=[0, 1], index_col=0)
        completed_seeds = {
            int(col.split("_")[1]) for col in existing_df.columns.get_level_values(0)
        }
        remaining_seeds = [seed for seed in seeds if seed not in completed_seeds]
        results_df = existing_df
    else:
        remaining_seeds = seeds
        # Initialize empty DataFrame with multi-level columns
        columns = pd.MultiIndex.from_tuples(
            [
                (f"seed_{seed}", metric)
                for seed in seeds
                for metric in ["x_index", "y_value"]
            ]
        )
        results_df = pd.DataFrame(index=range(total_iterations), columns=columns)
        results_df.index.name = "iteration"

    # Run optimization for remaining seeds
    for seed in tqdm(remaining_seeds, desc="Running optimization loops"):
        try:
            indices, values = run_single_loop(
                X=X, y=y, model_class=model_class, seed=seed, **kwargs
            )

            # Pad with NaN if necessary
            indices.extend([np.nan] * (total_iterations - len(indices)))
            values.extend([np.nan] * (total_iterations - len(values)))

            # Update DataFrame
            results_df[f"seed_{seed}", "x_index"] = indices
            results_df[f"seed_{seed}", "y_value"] = values

            # Save after each seed completion
            save_df = results_df.copy()
            if invert_y:
                # Invert only the y_value columns
                y_cols = [col for col in save_df.columns if col[1] == "y_value"]
                save_df[y_cols] = 1.0 / save_df[y_cols]
            save_df.to_csv(output_path)

        except Exception as e:
            print(f"Error in seed {seed}: {str(e)}")
            # Still save results even if there's an error
            results_df.to_csv(output_path)
            continue

    return results_df
