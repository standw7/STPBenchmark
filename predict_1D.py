import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from models import VarTGP, VarGP, ExactGP, VarLGP
from utils import set_seeds, TorchStandardScaler, TorchNormalizer
from optim import train_exact_model_botorch, train_natural_variational_model
from botorch.acquisition import LogExpectedImprovement

# Setup
set_seeds(4)


# Generate 1D synthetic data with outliers
def target_function(x):
    return np.sin(10 * x) * np.exp(-2 * x) + 0.2 * np.sin(25 * x)


# Generate data
n_train = 15
n_test = 200
# x_train_raw = torch.linspace(0, 1, n_train).unsqueeze(-1).double()
x_train_raw = torch.rand(n_train).unsqueeze(-1).double()
x_test_raw = torch.linspace(0, 1, n_test).unsqueeze(-1).double()

# Add noise and outliers to training data
y_train = torch.tensor([target_function(x.item()) for x in x_train_raw]).double()
noise = torch.randn(n_train).double() * 0.12
# Dynamic outlier indices
outlier_idx = [int(n_train / 4), int(n_train / 2), int(3 * n_train / 4)]
outliers = torch.randn(len(outlier_idx)).double()
for i, idx in enumerate(outlier_idx):
    noise[idx] = outliers[i]
y_train = y_train + noise

# True function values (for plotting)
y_true = torch.tensor([target_function(x.item()) for x in x_test_raw]).double()

# Normalize X data to [0,1] range and standardize Y data
x_normalizer = TorchNormalizer()
x_train = x_normalizer.fit_transform(x_train_raw)
x_test = x_normalizer.transform(x_test_raw)

y_scaler = TorchStandardScaler()
y_train_std = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()


# Function to train and evaluate models
def train_and_evaluate(model_class, model_name, x_train, y_train_std, x_test):
    # Create model
    if model_class == ExactGP:
        model = model_class(x_train, y_train_std).double()
        train_exact_model_botorch(model, x_train, y_train_std, y_standardize=False)
    else:
        inducing_points = x_train.clone()
        model = model_class(inducing_points).double()
        train_natural_variational_model(
            model, x_train, y_train_std, epochs=600, lr=0.01, y_standardize=False
        )

    # Get predictions
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model(x_test)
        mean = posterior.mean
        lower, upper = posterior.confidence_region()

        # Transform back to original scale
        mean = y_scaler.inverse_transform(mean.reshape(-1, 1)).flatten()
        lower = y_scaler.inverse_transform(lower.reshape(-1, 1)).flatten()
        upper = y_scaler.inverse_transform(upper.reshape(-1, 1)).flatten()

        # Calculate LogEI acquisition function
        best_f = y_train_std.max()
        with torch.no_grad():
            logei = LogExpectedImprovement(model, best_f=best_f, maximize=True)
            acq_values = logei(x_test.unsqueeze(1)).detach()
            if acq_values.dim() > 1 and acq_values.shape[0] > 1:
                acq_values = acq_values.mean(dim=0)

    return model, mean, lower, upper, acq_values


# Create list of models to evaluate
model_classes = [ExactGP, VarGP, VarTGP, VarLGP]
model_names = ["ExactGP", "VarGP", "VarTGP", "VarLGP"]
colors = ["#003f5c", "#7a5195", "#ef5675", "#ffa600"]

# Create figure with two subplots (means and acquisition functions)
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot the true function and training data on both axes
axes[0].plot(x_test_raw.flatten(), y_true, "k--", label="True Function", linewidth=3)
axes[0].plot(
    x_train_raw.flatten(),
    y_train,
    "k.",
    markersize=10,
    label="Training Data",
    zorder=10,
)

# Mark outliers
for idx in outlier_idx:
    axes[0].plot(
        x_train_raw[idx].item(),
        y_train[idx].item(),
        "ko",
        markersize=10,
        fillstyle="none",
        mew=2,
    )

# Store results for all models
means = []
acquisitions = []

# Train and plot each model
for i, (model_class, model_name, color) in enumerate(
    zip(model_classes, model_names, colors)
):
    print(f"Training {model_name}...")
    model, mean, lower, upper, acq_values = train_and_evaluate(
        model_class, model_name, x_train, y_train_std, x_test
    )

    # Store results
    means.append(mean)
    acquisitions.append(acq_values)

    # Plot mean and confidence intervals
    axes[0].plot(
        x_test_raw.flatten(),
        mean,
        "-",
        color=color,
        label=f"{model_name} Mean",
        linewidth=3,
    )
    axes[0].fill_between(x_test_raw.flatten(), lower, upper, alpha=0.15, color=color)

    # Print model parameters
    print(f"\n{model_name} Parameters:")
    if hasattr(model, "likelihood") and hasattr(model.likelihood, "noise"):
        print(f"  Noise parameter: {model.likelihood.noise.item():.4f}")
        if model_name == "VarTGP" and hasattr(model.likelihood, "deg_free"):
            print(f"  Degrees of freedom: {model.likelihood.deg_free.item():.4f}")

    if hasattr(model, "covar_module") and hasattr(model.covar_module, "outputscale"):
        print(f"  Output scale: {model.covar_module.outputscale.item():.4f}")

    if hasattr(model, "covar_module") and hasattr(
        model.covar_module.base_kernel, "lengthscale"
    ):
        if model.covar_module.base_kernel.lengthscale.numel() > 1:
            print(
                f"  Lengthscales: {model.covar_module.base_kernel.lengthscale.detach().numpy()}"
            )
        else:
            print(
                f"  Lengthscale: {model.covar_module.base_kernel.lengthscale.item():.4f}"
            )

# Plot acquisition functions
for i, (model_name, color, acq_values) in enumerate(
    zip(model_names, colors, acquisitions)
):
    # Normalize acquisition values for better comparison
    normalized_acq = (acq_values - acq_values.min()) / (
        acq_values.max() - acq_values.min()
    )
    axes[1].plot(
        x_test_raw.flatten(),
        normalized_acq,
        "-",
        color=color,
        label=f"{model_name} LogEI",
        linewidth=3,
    )
    # Plot the maximum acquisition value as a red dot
    max_idx = torch.argmax(acq_values)
    axes[1].plot(
        x_test_raw[max_idx].item(),
        normalized_acq[max_idx].item(),
        color=color,
        marker="d",
        ls="none",
        markersize=8,
    )


# Configure plots
axes[0].set_title("GP Model Predictions", fontsize=14)
axes[0].set_ylabel("f(x)", fontsize=12)
axes[0].legend()

axes[1].set_title("Log Expected Improvement Acquisition Function", fontsize=14)
axes[1].set_xlabel("x", fontsize=12)
axes[1].set_ylabel("Normalized LogEI", fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.savefig("gp_models_comparison.png", dpi=300)
plt.show()
