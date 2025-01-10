# run python file with ctrl+p
import os
import torch
import numpy as np
import pandas as pd
from models import VarSTP, VarGP, ExactGP
import matplotlib.pyplot as plt
from utils import TorchNormalizer
from runners import run_many_loops
from visualization import plot_optimization_trace, plot_top_values_discovery

datasets = os.listdir("data")  # pull in the benchmark sets
# model_classes = [VarSTP, VarGP, ExactGP]
model_classes = [ExactGP]
# model_names = ["VarSTP", "VarGP", "ExactGP"]
model_names = ["ExactGP"]
for model_name, model_class in zip(model_names, model_classes):
    for dataset in datasets:

        data = np.loadtxt(f"data/{dataset}", delimiter=",", skiprows=1)

        # average target of duplicate feature rows
        features, target = data[:, :-1], data[:, -1]
        unique_f, inv = np.unique(features, axis=0, return_inverse=True)
        data = np.column_stack(
            (unique_f, np.bincount(inv, weights=target) / np.bincount(inv))
        )

        X = torch.tensor(data[:, 0:-1], dtype=torch.double)
        y = torch.tensor(data[:, -1], dtype=torch.double).flatten()
        if dataset in ["Perovskite_dataset.csv", "AgNP_dataset.csv"]:
            print(f"\nInverting {dataset[:-4]} target values for maximization problem")
            y = 1.0 / y

        X = TorchNormalizer().fit_transform(X)

        seed_list = np.loadtxt("random_seeds.txt", dtype=int)

        print(f"\nRunning {model_name} on {dataset} dataset")

        results = run_many_loops(
            X,
            y,
            # seeds=seed_list[:3],
            seeds=[6185],
            n_initial=10,
            n_trials=90,
            epochs=200,
            learning_rate=0.05,
            model_class=model_class,
        )

        df = pd.DataFrame({key: value["y_selected"] for key, value in results.items()})
        if dataset in ["Perovskite_dataset.csv", "AgNP_dataset.csv"]:
            df = 1.0 / df

        df.to_csv(f"results/{model_name}_{dataset[:-4]}_traces.csv", index=False)
