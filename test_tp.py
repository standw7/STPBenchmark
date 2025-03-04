import jax.numpy as jnp
import jax.random as jra
import numpyro
import pytest
import numpy as np
from scipy import stats
from TProcess import *


def test_kernel_properties():
    """Test basic properties of the squared exponential kernel."""
    rng_key = jra.PRNGKey(0)
    X = jra.normal(rng_key, (10, 2))

    # Test kernel matrix properties
    K = squared_exponential_kernel(X, X, amplitude=1.0, length_scale=1.0)

    # Test symmetry
    assert jnp.allclose(K, K.T)

    # Test positive definiteness
    eigvals = jnp.linalg.eigvalsh(K)
    assert jnp.all(eigvals > -1e-10)  # Allow for numerical error

    # Test scaling with amplitude
    K2 = squared_exponential_kernel(X, X, amplitude=2.0, length_scale=1.0)
    assert jnp.allclose(K2, 4 * K)


def test_nu_scaling():
    """Test that the covariance scaling with nu is correct."""
    rng_key = jra.PRNGKey(0)
    X = jra.normal(rng_key, (10, 1))

    model = StudentTP(input_dim=1)

    # Test different nu values
    for nu in [3.0, 4.0, 5.0]:
        # Manually compute scaled kernel
        K = squared_exponential_kernel(X, X, amplitude=1.0, length_scale=1.0)
        K_scaled = ((nu - 2) / nu) * K

        # Get kernel from model
        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.condition(
                data={"nu": nu, "amplitude": 1.0, "length_scale": 1.0, "noise": 0.1}
            ):
                trace = numpyro.trace(model.model).get_trace(X)

        model_K = trace["y"]["scale_tril"] @ trace["y"]["scale_tril"].T

        assert jnp.allclose(K_scaled, model_K, rtol=1e-5)


def test_prediction_scaling():
    """Test that prediction scaling matches expected behavior."""
    rng_key = jra.PRNGKey(0)
    X_train = jnp.linspace(-5, 5, 20)[:, None]
    y_train = jnp.sin(X_train[:, 0])
    X_test = jnp.linspace(-6, 6, 50)[:, None]

    model = StudentTP(input_dim=1)

    # Fit model with fixed parameters for testing
    with numpyro.handlers.seed(rng_seed=0):
        model.fit(rng_key, X_train, y_train)

    # Get predictions
    means, samples = model.predict(rng_key, X_test)

    # Test shape of outputs
    assert means.shape[0] == len(samples)  # Number of posterior samples
    assert means.shape[1] == len(X_test)  # Number of test points

    # Verify that predictions become more uncertain far from training data
    var = jnp.var(samples, axis=0)
    assert jnp.mean(var[:5]) > jnp.mean(var[20:30])  # More variance at edges
    assert jnp.mean(var[-5:]) > jnp.mean(var[20:30])


def test_known_function():
    """Test model performance on a known function with noise."""
    rng_key = jra.PRNGKey(0)

    # Generate data from known function
    X = jnp.linspace(-3, 3, 30)[:, None]
    true_f = lambda x: jnp.sin(x) + 0.1 * x**2
    noise = 0.1
    y = true_f(X) + noise * jra.normal(rng_key, X.shape)

    # Fit model
    model = StudentTP(input_dim=1)
    model.fit(rng_key, X, y)

    # Make predictions
    X_test = jnp.linspace(-3, 3, 50)[:, None]
    means, samples = model.predict(rng_key, X_test)

    # Compute true values
    true_values = true_f(X_test)

    # Check prediction accuracy
    mean_predictions = jnp.mean(means, axis=0)
    rmse = jnp.sqrt(jnp.mean((mean_predictions - true_values.squeeze()) ** 2))
    assert rmse < 0.2  # Should be reasonably accurate

    # Check calibration of uncertainty estimates
    std_predictions = jnp.std(samples, axis=(0, 1))
    inside_1std = jnp.mean(
        jnp.abs(mean_predictions - true_values.squeeze()) < std_predictions
    )
    assert 0.6 < inside_1std < 0.9  # Roughly 68% should be within 1 std


def test_limiting_behavior():
    """Test that model approaches GP behavior as nu increases."""
    rng_key = jra.PRNGKey(0)
    X = jnp.linspace(-3, 3, 20)[:, None]
    y = jnp.sin(X[:, 0])

    # Fit models with different nu values
    predictions = {}
    for nu_prior in [3.0, 10.0, 100.0]:
        model = StudentTP(
            input_dim=1, nu_prior_dist=numpyro.distributions.Delta(nu_prior)
        )
        model.fit(rng_key, X, y)
        means, _ = model.predict(rng_key, X)
        predictions[nu_prior] = jnp.mean(means, axis=0)

    # As nu increases, predictions should converge
    diff_small = jnp.mean(jnp.abs(predictions[10.0] - predictions[3.0]))
    diff_large = jnp.mean(jnp.abs(predictions[100.0] - predictions[10.0]))
    assert diff_large < diff_small  # Convergence as nu increases


def test_edge_cases():
    """Test model behavior in edge cases."""
    rng_key = jra.PRNGKey(0)

    # Test with minimal number of points
    X_min = jra.normal(rng_key, (3, 1))
    y_min = jnp.ones(3)

    model = StudentTP(input_dim=1)
    model.fit(rng_key, X_min, y_min)

    # Test with repeated inputs
    X_repeat = jnp.array([[1.0], [1.0], [2.0]])
    y_repeat = jnp.array([1.0, 1.0, 2.0])

    model.fit(rng_key, X_repeat, y_repeat)

    # Test prediction at training points
    means, _ = model.predict(rng_key, X_repeat)
    assert jnp.allclose(jnp.mean(means, axis=0), y_repeat, rtol=0.1)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
