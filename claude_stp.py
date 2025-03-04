import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jra
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt


class StudentTProcess:
    def __init__(
        self,
        input_dim: int,
        nu: float = 3.0,
        lengthscale_prior_dist=None,
        amplitude_prior_dist=None,
        noise_prior_dist=None,
        include_noise: bool = True,
        mean_fn=None,
        jitter: float = 1e-9,
    ):
        """
        Student's T Process for regression.

        Args:
            input_dim: Dimensionality of the input space
            nu: Degrees of freedom (must be > 2)
            lengthscale_prior_dist: Prior distribution for kernel lengthscale
            amplitude_prior_dist: Prior distribution for kernel amplitude
            noise_prior_dist: Prior distribution for observation noise (if include_noise=True)
            include_noise: Whether to include observation noise in the model
            mean_fn: Mean function, defaults to zero mean
            jitter: Small constant added to the diagonal of kernel matrices for numerical stability
        """
        # Validate nu > 2 requirement (necessary for covariance to be defined)
        if nu <= 2:
            raise ValueError(f"Degrees of freedom (nu) must be > 2, got {nu}")

        self.input_dim = input_dim
        self.nu = nu
        self.jitter = jitter
        self.include_noise = include_noise
        self.lengthscale_prior_dist = lengthscale_prior_dist or dist.LogNormal(0, 1)
        self.amplitude_prior_dist = amplitude_prior_dist or dist.LogNormal(0, 1)
        self.noise_prior_dist = noise_prior_dist or dist.LogNormal(
            -3, 0.1
        )  # small noise
        self.mean_fn = mean_fn or (lambda x: jnp.zeros(x.shape[0]))

        self.X_train = None
        self.y_train = None
        self.mcmc = None

    def squared_exponential_kernel(self, X1, X2, lengthscale, amplitude):
        """Squared exponential (RBF) kernel function"""
        X1 = X1 / lengthscale
        X2 = X2 / lengthscale
        X1_norm = jnp.sum(X1**2, axis=1, keepdims=True)
        X2_norm = jnp.sum(X2**2, axis=1, keepdims=True)

        K = -2.0 * jnp.matmul(X1, X2.T) + X1_norm + X2_norm.T
        K = amplitude**2 * jnp.exp(-0.5 * K)

        return K

    def model(self, X, y=None):
        """Probabilistic model for the Student-t Process"""
        # Sample kernel hyperparameters
        with numpyro.plate("ard", self.input_dim):
            lengthscale = numpyro.sample("lengthscale", self.lengthscale_prior_dist)

        amplitude = numpyro.sample("amplitude", self.amplitude_prior_dist)

        # Sample noise if including it in the model
        noise = None
        if self.include_noise:
            noise = numpyro.sample("noise", self.noise_prior_dist)

        # Compute kernel matrix K11 (train-train)
        K11 = self.squared_exponential_kernel(X, X, lengthscale, amplitude)

        # Add noise to the diagonal if needed
        if self.include_noise:
            K11 = K11 + jnp.eye(X.shape[0]) * noise**2

        # Add jitter only to training kernel for numerical stability
        K11 = K11 + jnp.eye(X.shape[0]) * self.jitter

        # Compute mean function
        f_mean = self.mean_fn(X)

        # Define likelihood with proper scaling for Student-t
        K11_scaled = K11 * (self.nu - 2) / self.nu

        numpyro.sample(
            "y",
            dist.MultivariateStudentT(
                df=self.nu, loc=f_mean, scale_tril=jnp.linalg.cholesky(K11_scaled)
            ),
            obs=y,
        )

    def get_posterior(self, X_new, lengthscale, amplitude, noise=None):
        """Compute the posterior mean and covariance at new input locations"""
        # Compute kernel matrices
        K11 = self.squared_exponential_kernel(
            self.X_train, self.X_train, lengthscale, amplitude
        )
        K12 = self.squared_exponential_kernel(
            self.X_train, X_new, lengthscale, amplitude
        )
        K22 = self.squared_exponential_kernel(X_new, X_new, lengthscale, amplitude)

        # Add noise to training data kernel if needed
        if self.include_noise and noise is not None:
            K11 = K11 + jnp.eye(self.X_train.shape[0]) * noise**2

        # Add jitter only to the training kernel
        K11 = K11 + jnp.eye(self.X_train.shape[0]) * self.jitter

        # Compute mean function values
        f_mean_train = self.mean_fn(self.X_train)
        f_mean_test = self.mean_fn(X_new)

        # Center the observations using the mean function
        y_centered = self.y_train - f_mean_train

        # Compute Mahalanobis distance (β) using Cholesky decomposition for stability
        L = jnp.linalg.cholesky(K11)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y_centered))
        beta = jnp.dot(y_centered, alpha)

        # Compute posterior mean using Cholesky factor
        v = jnp.linalg.solve(L, K12)
        mean = f_mean_test + jnp.dot(v.T, jnp.linalg.solve(L.T, y_centered))

        # Compute posterior covariance with STP scaling factor
        scale_factor = (self.nu + beta - 2) / (self.nu + len(self.y_train) - 2)
        cov = scale_factor * (K22 - jnp.dot(v.T, v))

        return mean, cov

    def fit(
        self,
        rng_key,
        X,
        y,
        num_warmup=1000,
        num_samples=1000,
        num_chains=1,
        progress_bar=True,
    ):
        """Fit the model using MCMC"""
        self.X_train = X
        self.y_train = y

        kernel = NUTS(self.model)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )

        self.mcmc.run(rng_key, X, y)
        return self.mcmc.get_samples()

    def predict(self, rng_key, X_new, noiseless=False, n_samples=1):
        """Predict at new input locations with uncertainty"""
        samples = self.mcmc.get_samples()

        # Vectorize prediction over parameter samples
        if self.include_noise:
            noise_samples = samples["noise"]
            # If noiseless prediction is requested, set noise to zero
            noise_values = jnp.zeros_like(noise_samples) if noiseless else noise_samples

            pred_fn = lambda ls, amp, ns: self.get_posterior(X_new, ls, amp, ns)

            means, covs = jax.vmap(pred_fn)(
                samples["lengthscale"], samples["amplitude"], noise_values
            )
        else:
            pred_fn = lambda ls, amp: self.get_posterior(X_new, ls, amp)

            means, covs = jax.vmap(pred_fn)(
                samples["lengthscale"], samples["amplitude"]
            )

        # Average parameter samples for mean prediction
        mean_pred = jnp.mean(means, axis=0)

        # Sample from posterior predictive distribution
        def sample_posterior(key, mean, cov):
            df = self.nu + len(self.y_train)
            cov_stable = cov + jnp.eye(cov.shape[0]) * self.jitter
            L = jnp.linalg.cholesky(cov_stable)
            return dist.MultivariateStudentT(df=df, loc=mean, scale_tril=L).sample(
                key, sample_shape=(n_samples,)
            )

        keys = jax.random.split(rng_key, len(means))
        posterior_samples = jax.vmap(sample_posterior)(keys, means, covs)

        return mean_pred, posterior_samples

    def compute_predictive_intervals(
        self, rng_key, X_new, noiseless=False, credible_interval=0.9
    ):
        """Calculate mean and credible intervals"""
        samples = self.mcmc.get_samples()

        # Vectorize prediction over parameter samples
        if self.include_noise:
            noise_samples = samples["noise"]
            # If noiseless prediction is requested, set noise to zero
            noise_values = jnp.zeros_like(noise_samples) if noiseless else noise_samples

            pred_fn = lambda ls, amp, ns: self.get_posterior(X_new, ls, amp, ns)

            means, covs = jax.vmap(pred_fn)(
                samples["lengthscale"], samples["amplitude"], noise_values
            )
        else:
            pred_fn = lambda ls, amp: self.get_posterior(X_new, ls, amp)

            means, covs = jax.vmap(pred_fn)(
                samples["lengthscale"], samples["amplitude"]
            )

        # Generate posterior samples
        keys = jax.random.split(rng_key, len(means))

        def sample_from_mvt(key, mean, cov, n_samples=50):
            df = self.nu + len(self.y_train)
            cov_stable = cov + jnp.eye(cov.shape[0]) * self.jitter
            L = jnp.linalg.cholesky(cov_stable)
            return dist.MultivariateStudentT(df=df, loc=mean, scale_tril=L).sample(
                key, (n_samples,)
            )

        # Get samples for each posterior parameter set
        all_samples = jax.vmap(sample_from_mvt)(keys, means, covs)
        all_samples = all_samples.reshape(-1, X_new.shape[0])

        # Compute percentiles
        lower = jnp.percentile(all_samples, 100 * (1 - credible_interval) / 2, axis=0)
        upper = jnp.percentile(all_samples, 100 * (1 + credible_interval) / 2, axis=0)
        mean_pred = jnp.mean(means, axis=0)

        return mean_pred, lower, upper

    def sample_prior(self, rng_key, X, n_samples=1):
        """Sample from prior predictive distribution"""
        prior_predictive = Predictive(self.model, num_samples=n_samples)
        samples = prior_predictive(rng_key, X)
        return samples["y"]


def plot_stp_priors():
    # Set up the input space (1D for visualization)
    X = jnp.array([0])[:, None]

    # Create three models with different nu values
    # nu_values = [3.0, 5.0, 30.0]
    nu_values = [100.0, 5.0, 3.0]

    # Set up plot
    plt.figure(figsize=(6, 4))

    # Fix random seed for reproducibility
    master_key = jax.random.PRNGKey(0)

    # Generate and plot samples for each nu value
    for i, nu in enumerate(nu_values):
        # Create new model with the specific nu
        stp = StudentTProcess(
            input_dim=1,
            nu=nu,
            include_noise=True,  # Sample from the pure prior process
        )

        # Get a new random key
        key = jax.random.fold_in(master_key, i)

        # Draw 5 samples from the prior
        prior_samples = stp.sample_prior(key, X, n_samples=50000)

        bins = np.linspace(-10, 10, 100)
        plt.hist(
            prior_samples[:, 0],
            bins=bins,
            density=True,
            alpha=1.0,
            lw=2,  # make lines thickner
            histtype="step",
            stacked=True,
            fill=False,
            label=f"nu={nu}",
        )
    plt.yscale("log")

    plt.tight_layout()
    plt.legend()
    plt.show()


# Set random seed
key = jax.random.PRNGKey(1)

# Generate synthetic data with heavy-tailed noise
x = jnp.linspace(-1, 1, 20)[:, None]  # 15 points
true_fn = lambda x: jnp.sin(x)
true_y = true_fn(x)

# Generate heavy-tailed noise using Student-t distribution with df=2.5
noise_key = jax.random.fold_in(key, 0)
noise = jax.random.t(noise_key, df=2.5, shape=x.shape[0]) * 0.1
# Add 2 significant outliers
noise = noise.at[3].set(noise[3] + 1.0)
noise = noise.at[10].set(noise[10] - 1.0)

y = true_y.squeeze() + noise

# Create a test grid for predictions
x_test = jnp.linspace(-1, 1, 100)[:, None]

# Create RNG keys for fitting and prediction
fit_key = jax.random.fold_in(key, 1)
pred_key = jax.random.fold_in(key, 2)

# Fit the Student-T Process
stp = StudentTProcess(nu=3.0, input_dim=1, include_noise=True)
stp.fit(
    fit_key, x, y, num_warmup=1000, num_samples=500
)  # Reduced MCMC samples for speed

# Get predictive intervals
mean_stp, lower, upper = stp.compute_predictive_intervals(
    pred_key, x_test, noiseless=False
)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot true function
plt.plot(x_test, true_fn(x_test), "k--", label="True function")

# Plot data points
plt.scatter(x, y, c="k", marker="o", label="Training data")

# Plot STP prediction with uncertainty
plt.plot(x_test.squeeze(), mean_stp, "g-", label="STP mean")
plt.fill_between(
    x_test.squeeze(),
    lower,
    upper,
    alpha=0.3,
    color="g",
    label="STP 90% CI",
)
plt.title("Student-T Process with Outliers")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
