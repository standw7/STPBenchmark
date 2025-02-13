import jax
import jax.numpy as jnp
import jax.random as jra
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
import jax.random as random
from jax.scipy.special import gammaln

# preliz


def squared_exponential_kernel(
    xi,
    xj,
    amplitude: float,
    length_scale: float,
    jitter: float = 1e-8,
    noise: float = None,
):
    """
    Compute the squared exponential (RBF) kernel matrix for multidimensional inputs.
    """
    # Ensure inputs are 2D arrays
    xi = xi.reshape(-1, 1) if len(xi.shape) == 1 else xi
    xj = xj.reshape(-1, 1) if len(xj.shape) == 1 else xj

    # Handle vector-valued length scales
    if isinstance(length_scale, (jnp.ndarray, list, tuple)):
        length_scale = jnp.array(length_scale).reshape(1, 1, -1)

    diff = xi[:, None, :] - xj[None, :, :]
    scaled_diff = diff / length_scale
    r = jnp.sum(scaled_diff**2, axis=-1)
    K = jitter + amplitude**2 * jnp.exp(-0.5 * r)

    if noise is not None:
        K += noise**2 * jnp.eye(xi.shape[0])

    return K


class StudentTP:
    """
    Student's t Process implementation following Shah et al. and Tang et al.
    Properly implements the hierarchical construction with inverse Wishart process prior.
    """

    def __init__(
        self,
        input_dim: int,
        mean_fn=None,
        noise_prior_dist=None,
        lengthscale_prior_dist=None,
        amplitude_prior_dist=None,
        nu_prior_dist=None,
        jitter=1e-8,
    ):
        self.input_dim = input_dim
        self.mean_fn = mean_fn
        self.noise_prior_dist = noise_prior_dist or dist.LogNormal(0, 1)
        self.lengthscale_prior_dist = lengthscale_prior_dist or dist.LogNormal(0, 1)
        self.amplitude_prior_dist = amplitude_prior_dist or dist.LogNormal(0, 1)
        self.nu_prior_dist = nu_prior_dist or dist.Gamma(8, 2)

        self.jitter = jitter

        self.X_train = None
        self.y_train = None
        self.mcmc = None

    def model(self, X, y=None):
        """
        Hierarchical model implementing inverse Wishart process prior over GP kernel.
        """
        n = X.shape[0]

        # Sample hyperparameters
        amplitude = numpyro.sample("amplitude", self.amplitude_prior_dist)
        noise = numpyro.sample("noise", self.noise_prior_dist)
        nu = numpyro.sample("nu", self.nu_prior_dist)
        # Sample lengthscale per dimension
        with numpyro.plate("ard", self.input_dim):
            length_scale = numpyro.sample("length_scale", self.lengthscale_prior_dist)

        # Compute mean
        f_loc = jnp.zeros(n)
        if self.mean_fn is not None:
            f_loc += self.mean_fn(X)

        # Compute base kernel
        K = squared_exponential_kernel(
            X, X, amplitude, length_scale, noise=noise, jitter=self.jitter
        )
        K = ((nu - 2) / nu) * K

        # Sample y according to Student-t Process formula
        numpyro.sample(
            "y",
            dist.MultivariateStudentT(
                df=nu,
                loc=f_loc,
                scale_tril=K,
            ),
            obs=y,
        )

    def predict(
        self,
        rng_key,
        X_test,
        samples=None,
        num_draws=1,
    ):
        """
        Make predictions incorporating both the inverse Wishart process prior
        and proper covariance scaling.
        """
        X_test = X_test if X_test.ndim > 1 else X_test[:, None]

        if samples is None:
            samples = self.get_samples(chain_dim=False)

        # Prepare vmap arguments
        n_samples = len(samples["length_scale"])
        vmap_args = (
            jra.split(rng_key, n_samples),
            samples["amplitude"],
            samples["length_scale"],
            samples["noise"],
            samples["nu"],
        )

        def predict_single(rng_key, amplitude, length_scale, noise, nu):

            # Compute base kernels
            K_train = squared_exponential_kernel(
                self.X_train,
                self.X_train,
                amplitude,
                length_scale,
                noise=noise,
                jitter=self.jitter,
            )
            K_test = squared_exponential_kernel(
                X_test, X_test, amplitude, length_scale, jitter=self.jitter
            )
            K_cross = squared_exponential_kernel(
                X_test, self.X_train, amplitude, length_scale, jitter=self.jitter
            )

            K_train = ((nu - 2) / nu) * K_train
            K_test = ((nu - 2) / nu) * K_test
            K_cross = ((nu - 2) / nu) * K_cross

            # Compute predictive mean
            K_train_cho = jax.scipy.linalg.cho_factor(K_train)
            alpha = jax.scipy.linalg.cho_solve(K_train_cho, self.y_train)
            mean = K_cross @ alpha

            # Compute beta1 term for scale factor (Mahalanobis distance)
            beta1 = jnp.dot(self.y_train, alpha)

            # Compute posterior covariance
            v = jax.scipy.linalg.cho_solve(K_train_cho, K_cross.T)
            K_post = K_test - K_cross @ v

            # Scale covariance according to the papers' formulation
            scale_factor = (nu + beta1 - 2) / (nu + len(self.y_train) - 2)
            K_post = scale_factor * K_post

            # Updated degrees of freedom
            nu_post = nu + len(self.y_train)

            # Sample from multivariate Student-t
            K_post_cho = jax.scipy.linalg.cho_factor(
                K_post + self.jitter * jnp.eye(len(X_test))
            )
            samples = dist.MultivariateStudentT(
                df=nu_post, loc=mean, scale_tril=K_post_cho[0]
            ).sample(rng_key, (num_draws,))

            return mean, samples

        # Vectorize predictions over parameter samples
        batch_predict = jax.vmap(predict_single)
        means, samples = batch_predict(*vmap_args)

        return means, samples

    def get_samples(self, chain_dim=False):
        """Get posterior samples after running MCMC."""
        if self.mcmc is None:
            raise ValueError("No samples available. Run 'fit' method first.")
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    def fit(
        self,
        rng_key,
        X,
        y,
        num_warmup=1000,
        num_samples=1000,
        num_chains=2,
        chain_method="sequential",
        progress_bar=True,
        print_summary=True,
    ):
        """Run MCMC to infer the TP parameters."""
        X = X if X.ndim > 1 else X[:, None]
        y = y.squeeze()

        self.X_train = X
        self.y_train = y

        init_strategy = init_to_median(num_samples=10)
        kernel = NUTS(self.model, init_strategy=init_strategy)

        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
        )

        self.mcmc.run(rng_key, X, y)

        if print_summary:
            self.mcmc.print_summary()


class GaussianProcess:
    """
    Gaussian Process implementation with inverse gamma priors on hyperparameters.

    Args:
        input_dim: Number of input dimensions
        mean_fn: Optional mean function (defaults to zero)
        noise_prior_dist: Optional custom prior for noise
        lengthscale_prior_dist: Optional custom prior for lengthscale
        amplitude_prior_dist: Optional custom prior for amplitude
    """

    def __init__(
        self,
        input_dim: int,
        mean_fn=None,
        noise_prior_dist=None,
        lengthscale_prior_dist=None,
        amplitude_prior_dist=None,
    ):
        self.input_dim = input_dim
        self.mean_fn = mean_fn
        self.noise_prior_dist = noise_prior_dist or dist.LogNormal(0, 1)
        self.lengthscale_prior_dist = lengthscale_prior_dist or dist.LogNormal(0, 1)
        self.amplitude_prior_dist = amplitude_prior_dist or dist.LogNormal(0, 1)

        self.X_train = None
        self.y_train = None
        self.mcmc = None

    def model(self, X, y=None, jitter=1e-6):
        """GP probabilistic model."""
        n = X.shape[0]

        # Sample hyperparameters
        amplitude = numpyro.sample("amplitude", self.amplitude_prior_dist)
        noise = numpyro.sample("noise", self.noise_prior_dist)
        # length_scale = numpyro.sample("length_scale", self.lengthscale_prior_dist)
        # Sample lengthscale per dimension
        with numpyro.plate("ard", self.input_dim):
            length_scale = numpyro.sample("length_scale", self.lengthscale_prior_dist)

        # Compute mean
        f_loc = jnp.zeros(n)
        if self.mean_fn is not None:
            f_loc += self.mean_fn(X)

        # Compute kernel
        K = squared_exponential_kernel(
            X, X, amplitude, length_scale, noise=noise, jitter=jitter
        )

        # Sample y according to GP formula
        numpyro.sample(
            "y", dist.MultivariateNormal(loc=f_loc, covariance_matrix=K), obs=y
        )

    def fit(
        self,
        rng_key,
        X,
        y,
        num_warmup=1000,
        num_samples=1000,
        num_chains=2,
        chain_method="sequential",
        progress_bar=True,
        print_summary=True,
    ):
        """Run MCMC to infer the GP parameters."""
        X = X if X.ndim > 1 else X[:, None]
        y = y.squeeze()

        self.X_train = X
        self.y_train = y

        init_strategy = init_to_median(num_samples=10)
        kernel = NUTS(self.model, init_strategy=init_strategy)

        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
        )

        self.mcmc.run(rng_key, X, y)

        if print_summary:
            self.mcmc.print_summary()

    def get_samples(self, chain_dim=False):
        """Get posterior samples after running MCMC."""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    def predict(
        self,
        rng_key,
        X_test,
        samples=None,
        num_draws=1,
        noiseless=False,
    ):
        """
        Make predictions at X_test points using posterior samples.

        For 2000 MCMC samples and num_draws=10, this produces:
        - means: Array of 2000 mean predictions (one per MCMC sample)
        - samples: Array of shape (2000, 10, n_test_points) containing
                  10 draws from each parameter set's predictive distribution
        """
        X_test = X_test if X_test.ndim > 1 else X_test[:, None]

        if samples is None:
            samples = self.get_samples(chain_dim=False)

        # Prepare vmap arguments
        n_samples = len(samples["amplitude"])
        vmap_args = (
            jra.split(rng_key, n_samples),
            samples["amplitude"],
            samples["length_scale"],
            samples["noise"],
        )

        def predict_single(rng_key, amplitude, length_scale, noise):
            """For each MCMC parameter sample, draw num_draws samples from predictive distribution"""
            # Handle noise for prediction
            pred_noise = 0.0 if noiseless else noise

            # Compute kernel matrices
            K_train = squared_exponential_kernel(
                self.X_train, self.X_train, amplitude, length_scale, noise=noise
            )
            K_test = squared_exponential_kernel(
                X_test, X_test, amplitude, length_scale, noise=pred_noise
            )
            K_cross = squared_exponential_kernel(
                X_test, self.X_train, amplitude, length_scale
            )

            # Compute mean prediction using Cholesky for stability
            K_train_cho = jax.scipy.linalg.cho_factor(K_train)
            alpha = jax.scipy.linalg.cho_solve(K_train_cho, self.y_train)
            mean = K_cross @ alpha

            # Compute posterior covariance
            v = jax.scipy.linalg.cho_solve(K_train_cho, K_cross.T)
            cov = K_test - K_cross @ v

            # Sample from multivariate normal
            samples = dist.MultivariateNormal(
                loc=mean, covariance_matrix=cov + 1e-6 * jnp.eye(len(X_test))
            ).sample(rng_key, (num_draws,))

            return mean, samples

        # Vectorize predictions over parameter samples
        batch_predict = jax.vmap(predict_single)
        means, samples = batch_predict(*vmap_args)

        return means, samples


class CustomMultivariateStudentT(Distribution):
    support = constraints.real_vector
    arg_constraints = {
        "df": constraints.positive,
        "loc": constraints.real_vector,
        "shape_matrix": constraints.positive_definite,
    }

    def __init__(self, df, loc, shape_matrix, validate_args=None):
        self.df = df
        self.loc = loc
        # Scale matrix by (ν-2)/ν per paper's equation (10)
        self.shape_matrix = shape_matrix * ((df - 2) / df)
        batch_shape = jnp.broadcast_shapes(jnp.shape(df), jnp.shape(loc)[:-1])
        event_shape = jnp.shape(loc)[-1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        key1, key2 = random.split(key)
        dim = self.loc.shape[-1]

        # Sample from chi-square distribution
        chi2 = dist.Chi2(self.df).sample(key1, sample_shape)

        # Sample from multivariate normal
        gaussian = dist.MultivariateNormal(self.loc, self.shape_matrix).sample(
            key2, sample_shape
        )

        # Scale samples according to Student-T construction
        scaling = jnp.sqrt(self.df / chi2)
        scaling = jnp.expand_dims(scaling, axis=-1)
        return self.loc + scaling * (gaussian - self.loc)

    def log_prob(self, value):
        """
        Compute log probability according to equation (10) from the paper.
        """
        dim = self.loc.shape[-1]
        diff = value - self.loc

        # Compute Mahalanobis distance
        M = jnp.sum(diff * (jnp.linalg.solve(self.shape_matrix, diff)), axis=-1)

        # Log determinant of shape matrix
        half_log_det = jnp.linalg.slogdet(self.shape_matrix)[1] / 2.0

        # Equation (10) from paper in log space
        return (
            gammaln((self.df + dim) / 2.0)
            - gammaln(self.df / 2.0)
            - (dim / 2.0) * jnp.log(self.df * jnp.pi)
            - half_log_det
            - ((self.df + dim) / 2.0) * jnp.log1p(M / self.df)
        )

    @property
    def mean(self):
        return jnp.broadcast_to(
            jnp.where(self.df > 1, self.loc, jnp.inf),
            self.batch_shape + self.event_shape,
        )

    @property
    def variance(self):
        return jnp.broadcast_to(
            jnp.where(
                self.df > 2, self.shape_matrix * (self.df / (self.df - 2)), jnp.inf
            ),
            self.batch_shape + self.event_shape + self.event_shape,
        )
