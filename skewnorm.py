import math
import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

# Import GPyTorch components
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
from gpytorch.priors import Prior
from gpytorch.constraints import Positive

# Constants
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_LOG_2 = math.log(2.0)
_standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)


class SkewNormal(Distribution):
    """
    Skew-Normal distribution implementation for PyTorch.

    The skew-normal distribution extends the normal distribution with a shape parameter
    that controls the skewness of the distribution.

    Args:
        loc (Tensor): Location parameter (mean when shape=0)
        scale (Tensor): Scale parameter (must be positive)
        shape (Tensor): Shape parameter (controls skewness)
        validate_args (bool, optional): Whether to validate inputs. Default: None
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "shape": constraints.real,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc: Tensor, scale: Tensor, shape: Tensor, validate_args=None):
        # Convert inputs to tensors if needed
        if not isinstance(loc, Tensor):
            loc = torch.as_tensor(loc).to(torch.get_default_dtype())
        if not isinstance(scale, Tensor):
            scale = torch.as_tensor(scale).to(loc)
        if not isinstance(shape, Tensor):
            shape = torch.as_tensor(shape).to(loc)

        self.loc, self.scale, self.shape = broadcast_all(loc, scale, shape)

        batch_shape = self.loc.size()
        event_shape = torch.Size()  # Scalar distribution

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self.__class__.__new__(self.__class__) if _instance is None else _instance
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.shape = self.shape.expand(batch_shape)

        super(SkewNormal, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        delta = self.shape / torch.sqrt(1 + self.shape**2)
        return self.loc + self.scale * delta * _SQRT_2_OVER_PI

    @property
    def variance(self) -> Tensor:
        delta = self.shape / torch.sqrt(1 + self.shape**2)
        return self.scale**2 * (1 - 2 * delta**2 / math.pi)

    @property
    def stddev(self) -> Tensor:
        return self.variance.sqrt()

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value).to(self.loc)

        # Standardize the value
        z = (value - self.loc) / self.scale

        # Compute log PDF components
        log_phi_z = _standard_normal.log_prob(z)

        # Add clamp for numerical stability
        cdf_alpha_z = _standard_normal.cdf(self.shape * z)
        cdf_alpha_z = torch.clamp(
            cdf_alpha_z,
            min=torch.finfo(cdf_alpha_z.dtype).tiny,
            max=1.0 - torch.finfo(cdf_alpha_z.dtype).eps,
        )
        log_Phi_alpha_z = torch.log(cdf_alpha_z)

        # Combine components: log(2) - log(omega) + log(phi(z)) + log(Phi(alpha*z))
        return _LOG_2 - torch.log(self.scale) + log_phi_z + log_Phi_alpha_z

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)

        # Sample standard normals
        u0 = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        u1 = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)

        # Calculate delta
        delta = self.shape / torch.sqrt(1 + self.shape**2)
        # Add clamp for stability if delta gets extremely close to 1
        delta = torch.clamp(delta, max=1.0 - torch.finfo(delta.dtype).eps)

        # Ensure sqrt argument is non-negative
        sqrt_arg = torch.clamp(1 - delta**2, min=0.0)
        z = delta * torch.abs(u0) + torch.sqrt(sqrt_arg) * u1

        # Transform to the target skew-normal distribution
        return self.loc + self.scale * z


class SkewNormalLikelihood(_OneDimensionalLikelihood):
    """
    Skew-Normal likelihood for GPyTorch.

    This likelihood assumes that targets are distributed according to a skew-normal distribution.

    Args:
        batch_shape (torch.Size, optional): Batch shape. Default: torch.Size([])
        scale_prior (Prior, optional): Prior for scale parameter. Default: None
        scale_constraint (Constraint, optional): Constraint for scale parameter. Default: Positive()
        shape_prior (Prior, optional): Prior for shape parameter. Default: None
        shape_constraint (Constraint, optional): Constraint for shape parameter. Default: constraints.real
    """

    def __init__(
        self,
        batch_shape: torch.Size = torch.Size([]),
        scale_prior=None,
        scale_constraint=None,
        shape_prior=None,
        shape_constraint=None,
    ) -> None:
        super().__init__()

        # Default constraints
        if scale_constraint is None:
            scale_constraint = Positive()
        if shape_constraint is None:
            shape_constraint = constraints.real

        # Create parameters
        self.raw_scale = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        self.raw_shape = torch.nn.Parameter(torch.zeros(*batch_shape, 1))

        # Register constraints
        self.register_constraint("raw_scale", scale_constraint)
        if shape_constraint is not constraints.real:
            self.register_constraint("raw_shape", shape_constraint)
        else:
            self.raw_shape_constraint = shape_constraint

        # Initialize parameters
        self.initialize(scale=1.0)
        self.initialize(shape=0.0)

        # Register priors if provided
        if scale_prior is not None:
            self.register_prior(
                "scale_prior", scale_prior, self._get_scale_param, self._set_scale_param
            )
        if shape_prior is not None:
            self.register_prior(
                "shape_prior", shape_prior, self._get_shape_param, self._set_shape_param
            )

    @property
    def scale(self) -> Tensor:
        constraint = self.raw_scale_constraint
        if hasattr(constraint, "transform"):
            return constraint.transform(self.raw_scale)
        return self.raw_scale

    @scale.setter
    def scale(self, value: Tensor) -> None:
        self._set_scale_param(value)

    def _get_scale_param(self) -> Tensor:
        return self.scale

    def _set_scale_param(self, value: Tensor) -> None:
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value).to(self.raw_scale)

        constraint = self.raw_scale_constraint
        raw_value = value  # Default

        if hasattr(constraint, "inverse_transform"):
            raw_value = constraint.inverse_transform(value)

        try:
            self.raw_scale.data = raw_value.view_as(self.raw_scale)
        except RuntimeError:
            self.raw_scale.data = raw_value

    @property
    def shape(self) -> Tensor:
        constraint = self.raw_shape_constraint
        # Only apply transform for non-real constraints
        if constraint is constraints.real:
            return self.raw_shape
        elif hasattr(constraint, "transform"):
            return constraint.transform(self.raw_shape)
        return self.raw_shape

    @shape.setter
    def shape(self, value: Tensor) -> None:
        self._set_shape_param(value)

    def _get_shape_param(self) -> Tensor:
        return self.shape

    def _set_shape_param(self, value: Tensor) -> None:
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value).to(self.raw_shape)

        constraint = self.raw_shape_constraint
        raw_value = value  # Default for real constraint

        if constraint is not constraints.real and hasattr(
            constraint, "inverse_transform"
        ):
            raw_value = constraint.inverse_transform(value)

        try:
            self.raw_shape.data = raw_value.view_as(self.raw_shape)
        except RuntimeError:
            self.raw_shape.data = raw_value

    def forward(self, function_samples: Tensor, *args, **kwargs) -> SkewNormal:
        """
        Returns a SkewNormal distribution using the GP function values as the location parameter.

        Args:
            function_samples (Tensor): Values from the GP function (used as loc parameter)

        Returns:
            SkewNormal: Distribution with loc=function_samples, and scale/shape from the likelihood
        """
        loc = torch.as_tensor(function_samples).to(self.raw_scale)
        s = self.scale.to(loc)
        a = self.shape.to(loc)
        return SkewNormal(loc=loc, scale=s, shape=a)
