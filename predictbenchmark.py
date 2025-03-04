import torch
import gpytorch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import preprocess_data, TorchStandardScaler
from models import VarTGP, VarGP, ExactGP  # Ensure this file is in the same directory
from optim import train_exact_model_botorch, train_natural_variational_model

datasets = ['AgNP_dataset.csv', 'AutoAM_dataset.csv', 'CrossedBarrel_dataset.csv', 'P3HT_dataset.csv', 'Perovskite_dataset.csv']
epochs = [200, 400, 600, 800, 1000]
# get 15 seeds from the random_seeds.txt file 
with open("random_seeds.txt", "r") as f:
    random_seeds = f.readlines()
random_seeds = [int(seed.strip()) for seed in random_seeds]
seeds = random_seeds[:15]

priors = ["LogNormalDimScaling"]  # Add more priors here from LogNormalDimScaling, LogNormal, Gamma, Normal Prior


# Function to compute RMSE per dimension

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)  # MSE calculation


def train_and_evaluate_model(X, y, model_class, model_name, prior, epochs=800, lr=0.01):
    mse_scores = []
    learned_lengthscales = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=seed)
        X_train, y_train = X_train.float(), y_train.float()
        X_test, y_test = X_test.float(), y_test.float()

        # Assign prior
        if model_class in [VarTGP, VarGP]:
            model = model_class(X_train, lengthscale=prior)
        else:
            model = model_class(X_train, y_train)

        likelihood = model.likelihood

        # Train model
        if model_name in ["VarTGP", "VarGP"]:
            train_natural_variational_model(model, X_train, y_train, epochs, lr, y_standardize=False)
        else:
            train_exact_model_botorch(model, X_train, y_train)

        model.eval()
        likelihood.eval()

        # Get predictions
        with torch.no_grad():
            y_pred = likelihood(model(X_test)).mean.cpu().numpy()
            y_true = y_test.cpu().numpy()

        # Compute MSE
        mse = compute_mse(y_true, y_pred)
        mse_scores.append(mse)

        # Extract learned lengthscale
        lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        learned_lengthscales.append(lengthscale)

    return np.mean(mse_scores), np.array(learned_lengthscales)  # Return avg MSE and lengthscales

def run_predict_benchmark(datasets, priors, epochs, save=False):
    total_results = []
    for prior in priors:
        for epoch in epochs:
            for dataset in datasets:
                print(f"Running benchmark on {dataset} with prior {prior} and {epoch} epochs")
                X, y = preprocess_data(filepath=f"data/{dataset}")

                # Inverse transform if necessary
                if dataset in ["Perovskite_dataset.csv", "AgNP_dataset.csv"]:
                    y = 1.0 / y

                y = TorchStandardScaler().fit_transform(y)

                mse_results = {}

                for model_class, model_name in zip([VarTGP, VarGP, ExactGP], ["VarTGP", "VarGP", "ExactGP"]):
                    mse, _ = train_and_evaluate_model(X, y, model_class, model_name, prior, epoch)
                    mse_results[model_name] = mse

                # Store results
                mse_results["Dataset"] = dataset
                mse_results["Prior"] = prior
                mse_results["Epochs"] = epoch
                total_results.append(mse_results)

        # Convert to DataFrame
        mse_df = pd.DataFrame(total_results)

        if save:
            mse_df.to_csv(f"results/varying_epoch_log_normal_dim_scaled_mse_comparison.csv", index=False)
            print("Saved MSE results to results/mse_comparison.csv")

    return mse_df

def get_average_predict_scores(datasets):
    avg_scores = {}
    for prior in priors:
        for dataset in datasets:
            mse_df = pd.read_csv(f"results/{dataset.replace('.csv', f'{epochs}_epochs_mse_comparison.csv')}")
            avg_scores[dataset] = mse_df.mean()
    return avg_scores

# Run the prediction benchmark
total_mse_df = run_predict_benchmark(datasets, priors, epochs, save=True)
print(total_mse_df)

# sort the varying_epoch_log_normal_mse_comparison.csv file by epoch and dataset
df = pd.read_csv("results/varying_epoch_log_normal_dim_scaled_mse_comparison.csv")
df = df.sort_values(by=["Dataset", "Epochs"])
# save the sorted file
df.to_csv("results/varying_epoch_log_normal_dim_scaled_mse_comparison_sorted.csv", index=False)