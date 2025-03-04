import torch
import matplotlib.pyplot as plt
import gpytorch
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
from sklearn.model_selection import train_test_split
from utils import preprocess_data, TorchStandardScaler, set_seeds
from models import VarTGP, VarGP, ExactGP  # Ensure this file is in the same directory
from optim import train_exact_model_botorch, train_natural_variational_model
import math
import time

start_time = time.time()


def fit_predict(
    X_train, y_train, X_test, y_test, model_class, epochs=100, lr=0.01, **kwargs
):
    """
    Fits a GP model to training data and evaluates it on test data.

    Args:
        X_train (torch.Tensor): Training input data
        y_train (torch.Tensor): Training target data
        X_test (torch.Tensor): Test input data
        y_test (torch.Tensor): Test target data
        model_class: Class of the GP model (VarTGP, VarGP, or ExactGP)
        epochs (int): Number of training epochs for variational models
        lr (float): Learning rate for optimization
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        tuple: (predictions, ground_truth, mse_score)
    """
    # Ensure inputs are double tensors
    X_train = X_train.double()
    y_train = y_train.double().flatten()
    X_test = X_test.double()
    y_test = y_test.double().flatten()

    # Standardize y values for training
    y_scaler = TorchStandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    # Create model based on model_class
    if model_class == ExactGP:
        # For ExactGP model
        kwargs.update({"X_train": X_train, "y_train": y_train_scaled})
        model = model_class(**kwargs).double()
        train_exact_model_botorch(model, X_train, y_train_scaled)
    else:
        # For variational models
        if "inducing_points" not in kwargs:
            kwargs["inducing_points"] = X_train
        model = model_class(**kwargs).double()
        train_natural_variational_model(
            model, X_train, y_train_scaled, epochs=epochs, lr=lr, y_standardize=False
        )

    # Get predictions
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(X_test)
        predictions_scaled = posterior.mean

    # Inverse transform predictions back to original scale
    predictions = y_scaler.inverse_transform(predictions_scaled)

    # Calculate MAE
    mae = torch.mean(torch.abs(predictions - y_test))

    return predictions, y_test, mae.item()


def evaluate_model_across_datasets(
    model_class, n_seeds=10, test_size=0.2, epochs=100, lr=0.01, **model_kwargs
):
    """
    Evaluates a GP model across multiple datasets and random seeds.

    Args:
        datasets_dir (str): Directory containing datasets
        model_class: GP model class to evaluate
        n_seeds (int): Number of random seeds to evaluate
        test_size (float): Proportion of data for testing
        epochs (int): Training epochs for variational models
        lr (float): Learning rate
        **model_kwargs: Additional arguments for model constructor, including lengthscale priors

    Returns:
        dict: Results organized by dataset and seed
    """
    # Load seed list
    seeds = np.loadtxt("random_seeds.txt", dtype=int)[:n_seeds]

    # Get dataset files
    dataset_files = [f for f in os.listdir("data") if f.endswith(".csv")]

    # Results dictionary
    results = {}

    for dataset_file in dataset_files:
        dataset_path = os.path.join("data", dataset_file)
        dataset_name = os.path.splitext(dataset_file)[0]

        print(f"\nEvaluating dataset: {dataset_name}")
        X, y = preprocess_data(filepath=dataset_path)

        print(X.shape)

        # Check if this is a minimization dataset
        # invert_y = dataset_file in ["Perovskite_dataset.csv", "AgNP_dataset.csv"]
        # if invert_y:
        #     y = 1.0 / y

        # Store results for this dataset
        dataset_results = {
            "seeds": [],
            "mae_scores": [],
            "predictions": [],
            "ground_truth": [],
        }

        for seed_idx, seed in enumerate(seeds):
            set_seeds(seed)

            # Create train/test split
            indices = torch.randperm(len(X))
            test_size_count = int(test_size * len(X))

            train_indices = indices[test_size_count:]
            test_indices = indices[:test_size_count]

            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            # Fit and predict
            predictions, ground_truth, mae = fit_predict(
                X_train,
                y_train,
                X_test,
                y_test,
                model_class,
                epochs=epochs,
                lr=lr,
                **model_kwargs if model_class != ExactGP else {},
            )

            # Store results
            dataset_results["seeds"].append(seed)
            dataset_results["mae_scores"].append(mae)
            dataset_results["predictions"].append(predictions)
            dataset_results["ground_truth"].append(ground_truth)

        # Compute summary statistics for this dataset
        dataset_results["mean_mae"] = np.mean(dataset_results["mae_scores"])
        dataset_results["std_mae"] = np.std(dataset_results["mae_scores"])

        # Store in overall results
        results[dataset_name] = dataset_results

    return results


if __name__ == "__main__":

    n_seeds = 25

    model_classes = [VarTGP, VarGP, ExactGP]
    datasets = [f for f in os.listdir("data") if f.endswith(".csv")]

    # (models, datasets, seeds)
    result_db = np.zeros((len(model_classes), len(datasets), n_seeds))

    # Parameters
    epochs = 600
    lr = 0.01
    lengthscale_prior = gpytorch.priors.LogNormalPrior(0.0, 1.0)

    for i, model_class in enumerate(model_classes):
        print(f"Running model: {model_class.__name__}")

        # Run evaluation
        results = evaluate_model_across_datasets(
            model_class=model_class,
            n_seeds=n_seeds,
            test_size=0.2,
            epochs=epochs,
            lr=lr,
            # lengthscale_prior=lengthscale_prior,
        )

        # Store results in result_db
        for j, dataset_name in enumerate(datasets):
            dataset_key = os.path.splitext(dataset_name)[0]
            result_db[i, j, :] = results[dataset_key]["mae_scores"]

        print("elapsed time", (round((time.time() - start_time) / 60, 2)), "minutes")

    np.save("results/dimscaled_result_db.npy", result_db)

    for idx in range(len(datasets)):

        # Extract data for the selected dataset
        dataset_name = os.path.splitext(datasets[idx])[0]
        data_to_plot = [result_db[i, idx, :] for i in range(len(model_classes))]
        model_names = [model.__name__ for model in model_classes]

        # Create box plot
        plt.figure(figsize=(10, 6))
        box = plt.boxplot(data_to_plot, labels=model_names, showmeans=True)

        # Add labels and title
        plt.ylabel("Mean Absolute Error")
        plt.title(f"Model Performance Comparison on {dataset_name} Dataset")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()
