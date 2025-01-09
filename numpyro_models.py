# numpyro versioons of GP and tGP models

# a lot of code and conventions borrowed from the GPax library

import warnings
from typing import Callable, Dict, Optional, Tuple, Type, Union

import jax
import math
from jax import jit, vmap
import jaxlib
import jax.numpy as jnp
import jax.random as jra
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median, Predictive

from numpyro.util import enable_x64

enable_x64()

clear_cache = jax._src.dispatch.xla_primitive_callable.cache_clear


def add_jitter(x, jitter=1e-6):
    return x + jitter


def get_keys(seed: int = 0):
    """
    Simple wrapper for jax.random.split to get
    rng keys for model inference and prediction
    """
    rng_key_1, rng_key_2 = jax.random.split(jax.random.PRNGKey(seed))
    return rng_key_1, rng_key_2


def square_scaled_distance(
    X: jnp.ndarray, Z: jnp.ndarray, lengthscale: Union[jnp.ndarray, float] = 1.0
) -> jnp.ndarray:
    r"""
    Computes a square of scaled distance, :math:`\|\frac{X-Z}{l}\|^2`,
    between X and Z are vectors with :math:`n x num_features` dimensions
    """
    scaled_X = X / lengthscale
    scaled_Z = Z / lengthscale
    X2 = (scaled_X**2).sum(1, keepdims=True)
    Z2 = (scaled_Z**2).sum(1, keepdims=True)
    XZ = jnp.matmul(scaled_X, scaled_Z.T)
    r2 = X2 - 2 * XZ + Z2.T
    return r2.clip(0)


@jit
def RBFKernel(
    X: jnp.ndarray,
    Z: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    jitter: float = 1e-6,
    **kwargs
) -> jnp.ndarray:
    """
    Radial basis function kernel for use with separate likelihood.

    This version only includes the core RBF correlation structure and a small
    jitter term for numerical stability. Observation noise should be handled
    by a separate likelihood function.

    Args:
        X: 2D vector with (number of points, number of features) dimension
        Z: 2D vector with (number of points, number of features) dimension
        params: Dictionary with kernel hyperparameters 'kernel_length' and 'kernel_scale'
        jitter: Small constant added to diagonal for numerical stability

    Returns:
        Pure RBF kernel matrix between X and Z with minimal jitter
    """
    r2 = square_scaled_distance(X, Z, params["kernel_length"])
    k = params["kernel_scale"] * jnp.exp(-0.5 * r2)
    if X.shape == Z.shape:
        k += jitter * jnp.eye(X.shape[0])  # Only add minimal jitter for stability
    return k


class Exact_tGP:
    """
    Gaussian proess with student's t liklihood
    """

    def __init__(self, input_dim: int) -> None:
        clear_cache()

        self.input_dim = input_dim

    def model(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """
        Model definition for the tGP model
        """

        # kernel lengthscale prior
        kernel_length = numpyro.sample(
            "kernel_length",
            dist.LogNormal(
                loc=jnp.ones(self.input_dim) * math.sqrt(2)
                + math.log(self.input_dim) * 0.5,
                scale=jnp.ones(self.input_dim) * math.sqrt(3),
            ),
        )

        # kernel variance (output scale) prior
        kernel_scale = numpyro.sample("kernel_scale", dist.LogNormal(0.0, 1.0))

        # likelihood noise prior
        likelihood_scale = numpyro.sample("likelihood_scale", dist.LogNormal(-4.0, 1.0))

        # likelihood deg freedom prior
        likelihood_df = numpyro.sample("likelihood_df", dist.Gamma(2.0, 0.1))

        # compute kernel matrix
        K = RBFKernel(
            X, X, {"kernel_length": kernel_length, "kernel_scale": kernel_scale}
        )

        # sample the latent GP function
        f = numpyro.sample(
            "f",
            dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=K),
        )

        # sample the likelihood
        numpyro.sample(
            "obs", dist.StudentT(df=likelihood_df, loc=f, scale=likelihood_scale), obs=y
        )

    def fit(
        self,
        rng_key: jnp.array,
        X: jnp.ndarray,
        y: jnp.ndarray,
        num_warmup: int = 2000,
        num_samples: int = 2000,
        num_chains: int = 1,
        # chain_method: str = "sequential",
        progress_bar: bool = True,
        print_summary: bool = True,
        device: Type[jaxlib.xla_extension.Device] = None,
        **kwargs: float
    ) -> None:

        self.X_train = X
        self.y_train = y

        init_strategy = init_to_median(num_samples=10)
        kernel = NUTS(self.model, init_strategy=init_strategy)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            # chain_method=chain_method,
            progress_bar=progress_bar,
            jit_model_args=False,
        )
        self.mcmc.run(rng_key, X, y, **kwargs)
        if print_summary:
            self._print_summary()

    def _print_summary(self):
        samples = self._get_samples(1)
        numpyro.diagnostics.print_summary(samples)

    def _get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    def _predict(
        self,
        X_new: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        rng_key: jnp.ndarray,
        use_cholesky: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Modified prediction to properly capture both GP and likelihood uncertainty
        """
        # Split the RNG key for two sampling steps
        key_f, key_eps = jax.random.split(rng_key)

        # Compute kernel matrices
        K_XX = RBFKernel(self.X_train, self.X_train, params)
        K_pX = RBFKernel(X_new, self.X_train, params)
        K_pp = RBFKernel(X_new, X_new, params)

        # Compute predictive mean and covariance for latent f
        if use_cholesky:
            K_xx_cho = jax.scipy.linalg.cho_factor(K_XX)
            K = K_pp - jnp.matmul(K_pX, jax.scipy.linalg.cho_solve(K_xx_cho, K_pX.T))
            mean = jnp.matmul(K_pX, jax.scipy.linalg.cho_solve(K_xx_cho, self.y_train))
        else:
            K_xx_inv = jnp.linalg.inv(K_XX)
            K = K_pp - jnp.matmul(K_pX, jnp.matmul(K_xx_inv, K_pX.T))
            mean = jnp.matmul(K_pX, jnp.matmul(K_xx_inv, self.y_train))

        # Sample latent f from its posterior
        f_sample = mean + jnp.linalg.cholesky(K) @ jax.random.normal(
            key_f, shape=(X_new.shape[0],)
        )

        # Sample from Student's t likelihood
        sample = f_sample + params["likelihood_scale"] * jax.random.t(
            key_eps, params["likelihood_df"], shape=X_new.shape[:1]
        )

        return mean, sample

    def predict(
        self,
        rng_key: jnp.ndarray,
        X_new: jnp.ndarray,
        use_cholesky: bool = True,
        num_samples: Optional[int] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make predictions at new input locations using the full posterior distribution.
        This method vectorizes predictions across all posterior samples, giving us
        uncertainty estimates that account for parameter uncertainty.

        Args:
            X_new: New input locations (n_new x input_dim)
            rng_key: JAX random key for sampling
            use_cholesky: Whether to use Cholesky decomposition
            num_samples: Optional number of posterior samples to use (default: all)

        Returns:
            Tuple of (means, samples) where:
            - means has shape (n_posterior_samples, n_new)
            - samples has shape (n_posterior_samples, n_new)
        """
        # Get posterior samples
        posterior_samples = self._get_samples()

        # Optionally thin the posterior samples
        if num_samples is not None:
            # Select random subset of samples
            sample_idx = jax.random.randint(
                rng_key,
                shape=(num_samples,),
                minval=0,
                maxval=posterior_samples["kernel_scale"].shape[0],
            )
            posterior_samples = {k: v[sample_idx] for k, v in posterior_samples.items()}

        # Create random keys for each prediction
        rng_keys = jax.random.split(rng_key, posterior_samples["kernel_scale"].shape[0])

        # Vectorize _predict over posterior samples
        means, samples = jax.vmap(
            lambda params, key: self._predict(X_new, params, key, use_cholesky)
        )(posterior_samples, rng_keys)

        return means, samples
