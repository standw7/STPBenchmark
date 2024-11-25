import numpy as np
import torch
import random
import math
import pandas as pd


# set the seed for all random use
def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


class TorchStandardScaler:
    def fit(self, x):
        x = x.clone()
        # calculate mean and std of the tensor
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        x = x.clone()
        # standardize the tensor
        x -= self.mean
        x /= self.std + 1e-10
        return x

    def fit_transform(self, x):
        # copy the tensor as to not modify the original
        x = x.clone()
        # calculate mean and std of the tensor
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
        # standardize the tensor
        x -= self.mean
        x /= self.std + 1e-10
        return x


class TorchNormalizer:
    def fit(self, x):
        # calculate the maximum value and the minimum value of the tensor
        self.max = torch.max(x, dim=0).values
        self.min = torch.min(x, dim=0).values

    def transform(self, x):
        # normalize the tensor
        return (x.clone() - self.min) / (self.max - self.min)

    def fit_transform(self, x):
        # calculate the maximum value and the minimum value of the tensor
        self.max = torch.max(x, dim=0).values
        self.min = torch.min(x, dim=0).values
        # normalize the tensor
        return (x.clone() - self.min) / (self.max - self.min)


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


def get_initial_samples(y, n_samples, percentile=95):
    # Get the threshold value for the given percentile
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
