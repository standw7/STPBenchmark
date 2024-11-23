import numpy as np
import torch
import random
from tqdm import trange
import gpytorch


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


def initial_points(x, y):
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
