# file for models
import torch
import gpytorch
import math
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan


class VarSTP(gpytorch.models.ApproximateGP):
    """Variational Gaussian Process with Student-T likelihood for heavy-tailed observations.

    Uses variational inference with inducing points for scalability and a Student-T
    likelihood for robustness to outliers. Incorporates BoTorch-style dimensionality-aware
    priors for improved performance across different input dimensions.

    Note:
        Expects inputs to be normalized to [0,1] and outputs to be standardized.
        Uses fixed inducing point locations (learn_inducing_locations=False).
    """

    def __init__(self, inducing_points):
        """Initialize the model with inducing points.

        Args:
            inducing_points: Tensor of shape (num_inducing, input_dim) containing
                           the locations of inducing points.
        """
        # Set up sparse variational approximation
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,  # Learn inducing points
        )
        super(VarSTP, self).__init__(variational_strategy)

        # Mean and covariance setup with dimensionality-aware priors
        self.mean_module = ConstantMean()

        # BoTorch-style lengthscale prior that scales with input dimension
        input_dim = inducing_points.size(1)
        lengthscale_prior = gpytorch.priors.LogNormalPrior(
            loc=math.sqrt(2) + math.log(input_dim) * 0.5, scale=math.sqrt(3)
        )

        base_kernel = RBFKernel(
            ard_num_dims=input_dim,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(
                2.5e-2, transform=None, initial_value=lengthscale_prior.mode
            ),
        )

        # Kernel with outputscale prior
        outputscale_prior = LogNormalPrior(loc=0.0, scale=1.0)
        self.covar_module = ScaleKernel(
            base_kernel,
            outputscale_prior=outputscale_prior,
            outputscale_constraint=gpytorch.constraints.GreaterThan(
                1e-4, transform=None, initial_value=outputscale_prior.mode
            ),
        )

        # Student-T likelihood with priors on noise and degrees of freedom
        noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
        df_prior = gpytorch.priors.GammaPrior(2.0, 0.1)  # For degrees of freedom
        self.likelihood = gpytorch.likelihoods.StudentTLikelihood(
            noise_prior=noise_prior,
            deg_free_prior=df_prior,
            noise_constraint=GreaterThan(
                1e-4, transform=None, initial_value=noise_prior.mode
            ),
        )

    @property
    def num_outputs(self) -> int:
        """Number of output dimensions. Required for BoTorch compatibility."""
        return 1

    def forward(self, x):
        """Compute the prior distribution at input locations x."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(self, X, observation_noise=True, num_samples=2048, **kwargs):
        """Compute the posterior distribution at test points X.

        Args:
            X: Test locations
            observation_noise: Whether to include likelihood noise in predictions

        Returns:
            MultivariateNormal distribution at test locations
        """
        self.eval()
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_samples):
            posterior = self(X)
            if observation_noise:
                posterior = self.likelihood(posterior)
        return posterior


class VarGP(ApproximateGP):
    """Variational Gaussian Process with Gaussian likelihood.

    Uses variational inference with inducing points for scalability. Incorporates
    BoTorch-style dimensionality-aware priors and learns inducing point locations.

    Note:
        Expects inputs to be normalized to [0,1] and outputs to be standardized.
        Uses learnable inducing point locations (learn_inducing_locations=True).
    """

    def __init__(self, inducing_points):
        """Initialize model with inducing points.

        Args:
            inducing_points: Initial inducing point locations, shape (num_inducing, input_dim)
        """
        # Set up sparse variational approximation with learnable inducing points
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,  # Fixed inducing points
        )
        super(VarGP, self).__init__(variational_strategy)

        # Mean and likelihood setup with BoTorch-style priors
        self.mean_module = ConstantMean()

        # Setup kernel with dimensionality-aware priors
        input_dim = inducing_points.size(1)
        lengthscale_prior = LogNormalPrior(
            loc=math.sqrt(2) + math.log(input_dim) * 0.5, scale=math.sqrt(3)
        )

        base_kernel = RBFKernel(
            ard_num_dims=input_dim,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(
                2.5e-2, transform=None, initial_value=lengthscale_prior.mode
            ),
        )

        # Kernel with outputscale prior
        outputscale_prior = LogNormalPrior(loc=0.0, scale=1.0)
        self.covar_module = ScaleKernel(
            base_kernel,
            outputscale_prior=outputscale_prior,
            outputscale_constraint=GreaterThan(
                1e-4, transform=None, initial_value=outputscale_prior.mode
            ),
        )

        noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(
                1e-4,
                transform=None,
                initial_value=noise_prior.mode,
            ),
        )

    @property
    def num_outputs(self) -> int:
        """Number of output dimensions. Required for BoTorch compatibility."""
        return 1

    def forward(self, x):
        """Compute the prior distribution at input locations x."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(self, X, observation_noise=True, num_samples=2048, **kwargs):
        """Compute the posterior distribution at test points X.

        Args:
            X: Test locations
            observation_noise: Whether to include likelihood noise in predictions

        Returns:
            MultivariateNormal distribution at test locations
        """
        self.eval()
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_samples):
            posterior = self(X)
            if observation_noise:
                posterior = self.likelihood(posterior)
        return posterior


class ExactGP(gpytorch.models.ExactGP):
    """Exact GP regression with BoTorch-style priors.

    Uses exact inference (suitable for moderate dataset sizes) with dimensionality-aware
    priors from BoTorch. Includes priors on lengthscales, outputscale, and noise.

    Note:
        Expects inputs to be normalized to [0,1] and outputs to be standardized.
    """

    def __init__(self, x_train, y_train, likelihood):
        """Initialize model with training data.

        Args:
            x_train: Training inputs, shape (n_samples, n_features)
            y_train: Training targets, shape (n_samples,)
        """
        # # Setup likelihood with BoTorch-style noise prior
        # noise_prior = gpytorch.priors.LogNormalPrior(loc=-4.0, scale=1.0)
        # self.likelihood = GaussianLikelihood(
        #     noise_prior=noise_prior,
        #     noise_constraint=gpytorch.constraints.GreaterThan(
        #         1e-4,
        #         transform=None,
        #         initial_value=noise_prior.mode,
        #     ),
        # )

        super(ExactGP, self).__init__(x_train, y_train, likelihood)

        # Mean and covariance setup with dimensionality-aware priors
        self.mean_module = gpytorch.means.ConstantMean()

        input_dim = x_train.shape[1]

        lengthscale_prior = gpytorch.priors.LogNormalPrior(
            loc=math.sqrt(2) + math.log(input_dim) * 0.5, scale=math.sqrt(3)
        )

        base_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=input_dim,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=gpytorch.constraints.GreaterThan(
                2.5e-4, transform=None, initial_value=lengthscale_prior.mode
            ),
        )

        # Kernel with outputscale prior
        outputscale_prior = gpytorch.priors.LogNormalPrior(loc=0.0, scale=1.0)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_prior=outputscale_prior,
            outputscale_constraint=gpytorch.constraints.GreaterThan(
                1e-4, transform=None, initial_value=outputscale_prior.mode
            ),
        )

        self.num_outputs = 1

    def forward(self, x):
        """Compute the prior distribution at input locations x."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(self, X, observation_noise=True, num_samples=2048, **kwargs):
        """Compute the posterior distribution."""
        self.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if observation_noise:
                posterior = self.likelihood(self(X))
            else:
                posterior = self(X)
            return posterior