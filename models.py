# file for models
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import TrilNaturalVariationalDistribution


class VarSTP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = gpytorch.likelihoods.StudentTLikelihood()

    @property
    def num_outputs(self) -> int:
        """Required for BoTorch compatibility"""
        return 1

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(self, X, observation_noise=False, **kwargs):
        """Required for BoTorch compatibility"""
        self.eval()  # Make sure model is in eval mode
        with torch.no_grad():
            mvn = self(X)
            if observation_noise:
                mvn = self.likelihood(mvn)
        return mvn

    # # update the objective function with the new training data
    # def update_objective(self, train_y):
    #     self.objective_function = gpytorch.mlls.VariationalELBO(
    #         self.likelihood, self, num_data=train_y.numel()
    #     )


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train):
        # build the model using the ExactGP model from gpytorch
        super(ExactGP, self).__init__(x_train, y_train)

        # use a constant mean, this value can be learned from the dataset
        self.mean_module = gpytorch.means.ConstantMean()

        # automatically determine the number of dimensions for the ARD kernel
        num_dimensions = x_train.shape[1]

        # use a scaled RBF kernel, the ScaleKernel allows the kernel to learn a scale factor for the dataset
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=num_dimensions)
        )
        self.covar_module.base_kernel.raw_lengthscale.requires_grad = False
        self.likelihood = GaussianLikelihood(
            noise_prior=gpytorch.priors.HalfNormalPrior(0.01)
        )
        self.objective_function = ExactMarginalLogLikelihood(self.likelihood, self)

        # set the number of outputs
        self.num_outputs = 1

    def forward(self, x):
        # forward pass of the model

        # compute the mean and covariance of the model
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        # return the MultivariateNormal distribution of the mean and covariance
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VarGP(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(VarGP, self).__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=inducing_points.size(1)))
        self.likelihood = GaussianLikelihood()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def num_outputs(self) -> int:
        """Required for BoTorch compatibility"""
        return 1

    def posterior(self, X, observation_noise=False, **kwargs):
        """Required for BoTorch compatibility"""
        self.eval()  # Make sure model is in eval mode
        with torch.no_grad():
            mvn = self(X)
            if observation_noise:
                mvn = self.likelihood(mvn)
        return mvn
