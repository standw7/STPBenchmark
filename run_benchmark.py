import os
import torch
import numpy as np
import pandas as pd
from models import VarTGP, VarGP, VarLGP, ExactGP
from utils import TorchNormalizer, preprocess_data
from runners import run_many_loops

SMOKE_TEST = False  # run reduced benchmark for testing purposes

if SMOKE_TEST:
    print("\nRUNNING SMOKE TEST")

seed_list = np.loadtxt("random_seeds.txt", dtype=int)
datasets = os.listdir("data")  # pull in the benchmark sets

model_classes = [VarTGP, VarGP, VarLGP, ExactGP]
model_names = ["VarLGP", "ExactGP"]

for model_name, model_class in zip(model_names, model_classes):

    for dataset in datasets:

        X, y = preprocess_data(filepath=f"data/{dataset}")

        # invert y values for datasets that are minimization problems
        if dataset in ["Perovskite_dataset.csv", "AgNP_dataset.csv"]:
            y = 1.0 / y

        print(f"\nRunning {model_name} on {dataset} dataset")

        results = run_many_loops(
            X,
            y,
            model_class=model_class,
            seeds=seed_list[:3] if SMOKE_TEST else seed_list,
            n_initial=10,
            n_trials=30 if SMOKE_TEST else 80,
            epochs=200 if SMOKE_TEST else 600,
            learning_rate=0.01,
            output_path=f"results/{model_name}_{dataset[:-4]}_traces_{'SMOKE' if SMOKE_TEST else ''}.csv",
            invert_y=dataset in ["Perovskite_dataset.csv", "AgNP_dataset.csv"],
        )
