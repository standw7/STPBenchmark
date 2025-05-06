import numpy as np
import torch
import random
import math
import pandas as pd
from typing import Tuple, Optional, List
from torch.quasirandom import SobolEngine
from gpytorch.distributions import Distribution, MultivariateNormal


# set the seed for all random use
def set_seeds(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


class TorchStandardScaler:
    """A PyTorch implementation of StandardScaler that standardizes features by removing the mean and scaling to unit variance.

    The standardization is performed along dimension 0 (rows) while preserving dimensions through keepdim=True.
    Transformation is performed using the formula: z = (x - mean) / std
    """

    def fit(self, x):
        """Compute the mean and standard deviation of the input tensor to be used for subsequent scaling.

        Args:
            x (torch.Tensor): Input tensor to be fitted
        """
        x = x.clone()
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        """Transform the input tensor using the previously computed mean and standard deviation.

        Args:
            x (torch.Tensor): Input tensor to be transformed

        Returns:
            torch.Tensor: Transformed tensor with zero mean and unit variance
        """
        x = x.clone()
        x -= self.mean
        x /= self.std + 1e-10
        return x

    def fit_transform(self, x):
        """Fit the scaler and transform the input tensor in one step.

        Args:
            x (torch.Tensor): Input tensor to be fitted and transformed

        Returns:
            torch.Tensor: Transformed tensor with zero mean and unit variance
        """
        x = x.clone()
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
        x -= self.mean
        x /= self.std + 1e-10
        return x

    def inverse_transform(self, x):
        """Transform standardized data back to the original scale.

        Args:
            x (torch.Tensor): Standardized input tensor

        Returns:
            torch.Tensor: Tensor transformed back to original scale
        """
        x = x.clone()
        x *= self.std + 1e-10
        x += self.mean
        return x


class TorchNormalizer:
    """A PyTorch implementation of MinMaxScaler that scales features to a fixed range [0, 1].

    The normalization is performed along dimension 0 (rows) while preserving the tensor dimensions.
    Transformation is performed using the formula: z = (x - min) / (max - min)
    """

    def fit(self, x):
        """Compute the minimum and maximum values of the input tensor to be used for subsequent scaling.

        Args:
            x (torch.Tensor): Input tensor to be fitted
        """
        self.max = torch.max(x, dim=0).values
        self.min = torch.min(x, dim=0).values

    def transform(self, x):
        """Transform the input tensor using the previously computed minimum and maximum values.

        Args:
            x (torch.Tensor): Input tensor to be transformed

        Returns:
            torch.Tensor: Transformed tensor with values scaled to [0, 1]
        """
        return (x.clone() - self.min) / (self.max - self.min)

    def fit_transform(self, x):
        """Fit the normalizer and transform the input tensor in one step.

        Args:
            x (torch.Tensor): Input tensor to be fitted and transformed

        Returns:
            torch.Tensor: Transformed tensor with values scaled to [0, 1]
        """
        self.max = torch.max(x, dim=0).values
        self.min = torch.min(x, dim=0).values
        return (x.clone() - self.min) / (self.max - self.min)

    def inverse_transform(self, x):
        """Transform normalized data back to the original scale.

        Args:
            x (torch.Tensor): Normalized input tensor

        Returns:
            torch.Tensor: Tensor transformed back to original scale
        """
        x = x.clone()
        x *= self.max - self.min
        x += self.min
        return x


def initial_points(x, y, num_initial_points):
    # Calculate the threshold for the top 5%
    top_5_percent_threshold = torch.quantile(y, 0.05)

    # Identify indices of samples not in the top 5%
    non_top_5_indices = (y > top_5_percent_threshold).nonzero(as_tuple=True)[0]

    # Randomly sample from these non-top-5% indices
    indicesSTP = torch.randperm(len(non_top_5_indices))[:num_initial_points]
    indicesSTP = non_top_5_indices[indicesSTP]

    # Select the corresponding points from train_x and train_y
    x_initial = x[indicesSTP]
    y_initial = y[indicesSTP]

    return x_initial, y_initial


def preprocess_data(
    filepath: str, skip_rows: int = 1, delimiter: str = ","
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess data from CSV file for benchmark experiments.

    Args:
        filepath: Path to CSV data file
        invert_target: Whether to invert target values (for minimization problems)
        skip_rows: Number of header rows to skip
        delimiter: CSV delimiter character

    Returns:
        Tuple of (features, targets) as normalized torch tensors

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data is empty or malformed
    """
    try:
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skip_rows)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except ValueError:
        raise ValueError(f"Failed to parse data from: {filepath}")

    if data.size == 0:
        raise ValueError("Empty dataset")

    # Split features and target
    features, target = data[:, :-1], data[:, -1]

    # Average target values for duplicate feature rows
    unique_features, inverse_indices = np.unique(features, axis=0, return_inverse=True)
    averaged_targets = np.bincount(inverse_indices, weights=target) / np.bincount(
        inverse_indices
    )

    # Combine unique features with averaged targets
    processed_data = np.column_stack((unique_features, averaged_targets))

    # Convert to tensors
    X = torch.tensor(processed_data[:, :-1], dtype=torch.double)
    y = torch.tensor(processed_data[:, -1], dtype=torch.double).flatten()

    # Normalize features
    X = TorchNormalizer().fit_transform(X)

    return X, y


def get_initial_samples(
    y: torch.Tensor, n_samples: int, percentile: float = 95
) -> torch.Tensor:
    """Randomly samples indices from values below a percentile threshold in a PyTorch tensor.

    Args:
        y: Input tensor of values to sample from
        n_samples: Number of indices to sample
        percentile: Upper percentile threshold for eligible values (default: 95)

    Returns:
        Tensor of randomly sampled indices where values are below the percentile threshold

    Raises:
        ValueError: If requested sample size exceeds number of eligible values
    """
    threshold = torch.quantile(y, percentile / 100)

    # Get indices where y is below the threshold
    eligible_indices = torch.where(y <= threshold)[0]

    # Randomly sample n indices from eligible ones
    if n_samples > len(eligible_indices):
        raise ValueError(
            f"Cannot sample {n_samples} points. Only {len(eligible_indices)} points below {percentile}th percentile."
        )

    # Random sampling in PyTorch
    perm = torch.randperm(len(eligible_indices))[:n_samples]
    sampled_indices = eligible_indices[perm]

    return sampled_indices


def get_initial_samples_sobol(
    X: torch.Tensor,
    y: torch.Tensor,
    n_samples: int = 10,
    percentile: float = 95.0,
) -> torch.Tensor:
    """
    Sample points using Sobol sequence from data below percentile threshold.
    """
    # Calculate threshold based on percentile
    threshold = torch.quantile(y, percentile / 100)

    # Find points below the threshold
    valid_indices = (y <= threshold).nonzero(as_tuple=True)[0]

    if len(valid_indices) < n_samples:
        return valid_indices

    # Get the valid points
    X_valid = X[valid_indices]

    # Normalize the valid points to [0,1] for proper scaling
    X_min = X_valid.min(dim=0)[0]
    X_max = X_valid.max(dim=0)[0]
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0  # Avoid division by zero

    # Generate Sobol points in unit hypercube
    d = X.shape[1]
    sobol_engine = SobolEngine(dimension=d, scramble=True)
    sobol_points = sobol_engine.draw(n_samples)

    # Scale Sobol points to the range of valid data
    scaled_sobol = sobol_points * X_range + X_min

    # Find nearest neighbors to scaled Sobol points
    selected_indices = []
    for sobol_point in scaled_sobol:
        distances = torch.norm(X_valid - sobol_point, dim=1)
        nearest_idx = valid_indices[torch.argmin(distances)]
        selected_indices.append(nearest_idx.item())

    return torch.tensor(selected_indices, dtype=torch.long)


# Set the device to use
def get_device():
    if torch.backends.mps.is_available():
        print("MPS is available")
        return torch.device("mps")

    elif torch.cuda.is_available():
        print("CUDA is available")
        return torch.device("cuda")
    else:
        print("CUDA/MPS is not available")
        return torch.device("cpu")


# Expected Improvement
def EI(mean, std, best_observed, minimize=False):
    if minimize:
        improvement = best_observed - mean
    else:
        improvement = mean - best_observed

    z = improvement / std
    normal = torch.distributions.Normal(0, 1)

    ei = improvement * normal.cdf(z) + std * normal.log_prob(z).exp()
    return ei


def get_top_samples(train_y, top_percentage=5):
    """
    Find the top percentage of samples from a given dataset.

    Args:
        train_y (torch.Tensor): Tensor containing the target values (e.g., labels or scores).
        top_percentage (float): Percentage of samples to extract as top samples (default is 5%).

    Returns:
        list: A list of the top percentage samples.
    """
    n_top = int(math.ceil(len(train_y) * (top_percentage / 100)))
    train_y_df = pd.DataFrame(train_y.numpy(), columns=[0])
    top_samples = (
        train_y_df.nlargest(n_top, train_y_df.columns[0], keep="first")
        .iloc[:, 0]
        .values.tolist()
    )

    print(f"Number of top {top_percentage}% samples: {len(top_samples)}")
    print(f"Top {top_percentage}% samples: {top_samples}")

    return top_samples


def num_top_samples(y, top_samples):
    return len([i for i in y if i in top_samples]) / len(top_samples)


# Function to dynamically collect arrays
def collect_arrays(prefix, num_arrays):
    arrays = []
    for i in range(num_arrays):
        array = globals().get(f"{prefix}{i}", None)
        if array is not None:
            arrays.append(array)
    return arrays


# Function to pad arrays with the last element to match the maximum length
def pad_array(array, max_length):
    return np.pad(
        array, (0, max_length - len(array)), "constant", constant_values=array[-1]
    )


def find_max_length(prefix, num_arrays):
    arrays = collect_arrays(prefix, num_arrays)
    return max(len(arr) for arr in arrays)


# Process arrays for each type
def process_arrays(prefix, num_arrays, max_length):
    arrays = collect_arrays(prefix, num_arrays)
    padded_arrays = [pad_array(arr, max_length) for arr in arrays]
    stack = np.stack(padded_arrays)
    mean_values = np.mean(stack, axis=0)
    std_values = np.std(stack, axis=0)
    return mean_values, std_values


def get_prediction_summary(
    posterior: Distribution,
    num_samples: int = 1024,
    quantiles: Tuple[float, float] = (0.025, 0.975),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute predictive mean, variance, and quantiles from a GPyTorch posterior.

    Args:
        posterior: GPyTorch posterior distribution (e.g., from model.posterior()).
        num_samples: Number of samples for Monte Carlo estimation (if needed).
        quantiles: Tuple of lower and upper quantiles (e.g., 0.025 and 0.975).

    Returns:
        Tuple of mean, variance, lower quantile, upper quantile (all torch.Tensor),
        each of shape [event_size].
    """

    likelihood_dist = posterior.__str__().split("(")[0]
    print(f"{likelihood_dist} likelihood detected, sampling accordingly.")

    if isinstance(posterior, MultivariateNormal):
        mean = posterior.mean
        variance = posterior.variance
        stddev = variance.clamp_min(1e-9).sqrt()
        normal = torch.distributions.Normal(0.0, 1.0)
        lower_ci = mean + stddev * normal.icdf(
            torch.tensor(quantiles[0], device=mean.device)
        )
        upper_ci = mean + stddev * normal.icdf(
            torch.tensor(quantiles[1], device=mean.device)
        )
    else:
        samples = posterior.sample(torch.Size([num_samples]))  # [S, B, E] or [S, E]
        if samples.dim() > 2:
            # Flatten [S, B, E] -> [S * B, E]
            samples = samples.view(-1, samples.shape[-1])
        else:
            # Already [S, E]
            pass
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)
        lower_ci = torch.quantile(samples, quantiles[0], dim=0)
        upper_ci = torch.quantile(samples, quantiles[1], dim=0)

    return mean.detach(), variance.detach(), lower_ci.detach(), upper_ci.detach()
