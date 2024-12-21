# custom acquisition functions for gpytorch models
import numpy as np
import torch
from scipy.stats import norm, t


def log_expected_improvement_t(
    model, test_X: torch.Tensor, best_f: torch.Tensor
) -> torch.Tensor:
    """Computes the log expected improvement acquisition function value using Student's t distribution.

    Args:
        model: A trained model that implements a posterior() method returning mean, variance, and df
        test_X: A tensor of points to evaluate the acquisition function
        best_f: Best observed value

    Returns:
        log_ei: A tensor containing the log expected improvement values
    """
    with torch.no_grad():
        posterior = model.posterior(test_X)
        mean = posterior.mean
        std = posterior.variance.sqrt()
        df = posterior.df

    if len(mean.shape) > 1:
        mean = torch.mean(mean, axis=0)
        std = torch.mean(std, axis=0)
        df = torch.mean(df, axis=0)

    # Convert tensors to numpy for scipy operations
    mean_np = mean.detach().cpu().numpy()
    std_np = std.detach().cpu().numpy()
    best_f_np = best_f.detach().cpu().numpy()
    df_np = df.detach().cpu().numpy()

    improvement = mean_np - best_f_np
    Z = improvement / std_np

    t_dist = t(df=df_np)
    log_pdf = t_dist.logpdf(Z)
    log_cdf = t_dist.logcdf(Z)

    positive_mask = improvement > 0
    logEI = np.where(
        positive_mask, np.log(improvement) + log_cdf, log_pdf + np.log(std_np)
    )

    eps = 1e-9
    logEI = np.where(std_np > eps, logEI, -np.inf)

    return torch.tensor(logEI, device=test_X.device, dtype=test_X.dtype)


def log_expected_improvement(
    model, test_X: torch.Tensor, best_f: torch.Tensor
) -> torch.Tensor:
    """Computes the log expected improvement acquisition function value.

    Args:
        model: A trained model that implements a posterior() method returning mean and variance
        test_X: A tensor of points to evaluate the acquisition function
        train_y: Training targets for computing the best observed value

    Returns:
        log_ei: A tensor containing the log expected improvement values

    Note:
        The model is assumed to be in evaluation mode before calling this function.
    """
    with torch.no_grad():
        posterior = model.posterior(test_X)
        mean = posterior.mean
        std = posterior.variance.sqrt()

    # Handle MC Likelihood Samples
    if len(mean.shape) > 1:
        mean = torch.mean(mean, axis=0)
        std = torch.mean(std, axis=0)

    eps = 1e-9
    std = std + eps

    improvement = mean - best_f
    Z = improvement / std

    norm = torch.distributions.Normal(0, 1)

    # Handle positive and negative improvements separately
    positive_improvement_mask = improvement > 0
    safe_improvement = torch.where(
        positive_improvement_mask, improvement, torch.ones_like(improvement)
    )

    # Compute components safely
    log_phi = norm.log_prob(Z)
    log_improvement = torch.log(safe_improvement)
    Phi = norm.cdf(Z)
    log_Phi = torch.log(Phi + eps)

    # Combine the components
    logEI = torch.where(
        positive_improvement_mask, log_improvement + log_Phi, log_phi + torch.log(std)
    )

    # Handle edge cases
    logEI = torch.where(std > eps, logEI, torch.zeros_like(improvement))

    return logEI
