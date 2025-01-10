# run python file with ctrl+p
import os
import torch
import numpy as np
import pandas as pd
from models import VarTGP, VarGP, ExactGP
import matplotlib.pyplot as plt
from utils import TorchNormalizer
from runners import run_many_loops
from visualization import plot_optimization_trace, plot_top_values_discovery

seed_list = np.loadtxt("random_seeds.txt", dtype=int)
datasets = os.listdir("data")  # pull in the benchmark sets

model_classes = [VarTGP, VarGP, ExactGP]
model_names = ["VarTGP", "VarGP", "ExactGP"]

for model_name, model_class in zip(model_names, model_classes):

    for dataset in datasets:

        data = np.loadtxt(f"data/{dataset}", delimiter=",", skiprows=1)

        # average target values of duplicate feature rows
        features, target = data[:, :-1], data[:, -1]
        unique_f, inv = np.unique(features, axis=0, return_inverse=True)
        data = np.column_stack(
            (unique_f, np.bincount(inv, weights=target) / np.bincount(inv))
        )

        # convert data to tensors
        X = torch.tensor(data[:, 0:-1], dtype=torch.double)
        y = torch.tensor(data[:, -1], dtype=torch.double).flatten()

        # normalize X values
        X = TorchNormalizer().fit_transform(X)

        # invert y values for datasets that are minimization problems
        if dataset in ["Perovskite_dataset.csv", "AgNP_dataset.csv"]:
            print(
                f"[INFO] Inverting {dataset[:-4]} target values for maximization problem"
            )
            y = 1.0 / y

        print(f"\nRunning {model_name} on {dataset} dataset")

        results = run_many_loops(
            X,
            y,
            model_class=model_class,
            seeds=seed_list[:3],
            n_initial=10,
            n_trials=90,
            epochs=300,
            learning_rate=0.05,
        )

        df = pd.DataFrame({key: value["y_selected"] for key, value in results.items()})

        # reverse the inverted y values for minimization datasets
        if dataset in ["Perovskite_dataset.csv", "AgNP_dataset.csv"]:
            df = 1.0 / df

        df.to_csv(f"results/{model_name}_{dataset[:-4]}_traces.csv", index=False)
