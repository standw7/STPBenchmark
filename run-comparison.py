import numpy as np
import matplotlib.pyplot as plt
from CModels import (
    StudentTProcess,
    GaussianProcess,
    six_hump_camel,
    run_bayesian_optimization,
    latin_hypercube_sampling,
)
from tqdm import trange


def test_six_hump_camel_function():
    """Test the implementation of the six-hump camel function"""
    # Known global minima
    min1 = six_hump_camel(0.0898, -0.7126)
    min2 = six_hump_camel(-0.0898, 0.7126)

    print(f"Global minimum 1: {min1}")
    print(f"Global minimum 2: {min2}")

    # Plot the function
    x1 = np.linspace(-3, 3, 300)
    x2 = np.linspace(-2, 2, 300)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros(X1.shape)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = six_hump_camel(X1[i, j], X2[i, j])

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X1, X2, Z, 50, cmap="viridis")
    plt.colorbar(contour)
    plt.title("Six-Hump Camel Function")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter([0.0898, -0.0898], [-0.7126, 0.7126], color="red", marker="x", s=100)
    plt.tight_layout()
    plt.savefig("six_hump_camel.png")
    plt.close()


def compare_processes_single_run():
    """Compare GP and STP on a single run"""

    # Define the function to optimize
    def func(x1, x2):
        return six_hump_camel(x1, x2)

    # Define bounds
    bounds = [(-3, 3), (-2, 2)]

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run Bayesian optimization with GP
    print("Running Bayesian optimization with GP...")
    results_gp = run_bayesian_optimization(
        func, bounds, n_init=20, n_iter=100, model_type="gp"
    )

    # Run Bayesian optimization with STP (nu=5)
    print("Running Bayesian optimization with STP (nu=5)...")
    results_stp5 = run_bayesian_optimization(
        func, bounds, n_init=20, n_iter=100, model_type="stp", nu=5
    )

    # Run Bayesian optimization with STP (nu=11)
    print("Running Bayesian optimization with STP (nu=11)...")
    results_stp11 = run_bayesian_optimization(
        func, bounds, n_init=20, n_iter=100, model_type="stp", nu=11
    )

    # Print results
    print(f"Best value found by GP: {results_gp['best_y']} at {results_gp['best_x']}")
    print(
        f"Best value found by STP (nu=5): {results_stp5['best_y']} at {results_stp5['best_x']}"
    )
    print(
        f"Best value found by STP (nu=11): {results_stp11['best_y']} at {results_stp11['best_x']}"
    )

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.log10(np.abs(np.array(results_gp["best_y_history"]) - (-1.0316))),
        "r-",
        label="GP",
    )
    plt.plot(
        np.log10(np.abs(np.array(results_stp5["best_y_history"]) - (-1.0316))),
        "g-",
        label="STP (nu=5)",
    )
    plt.plot(
        np.log10(np.abs(np.array(results_stp11["best_y_history"]) - (-1.0316))),
        "b-",
        label="STP (nu=11)",
    )
    plt.xlabel("Iteration")
    plt.ylabel("log10(|y_best - y*|)")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("convergence_comparison_single.png")
    plt.close()

    # Plot final evaluations
    plt.figure(figsize=(15, 5))

    # Create the function surface for background
    x1 = np.linspace(-3, 3, 300)
    x2 = np.linspace(-2, 2, 300)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros(X1.shape)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = six_hump_camel(X1[i, j], X2[i, j])

    # GP
    plt.subplot(1, 3, 1)
    contour = plt.contourf(X1, X2, Z, 50, cmap="viridis", alpha=0.7)
    plt.scatter(
        results_gp["X"][:20, 0],
        results_gp["X"][:20, 1],
        c="white",
        marker="o",
        label="Initial",
    )
    plt.scatter(
        results_gp["X"][20:, 0],
        results_gp["X"][20:, 1],
        c="red",
        marker="x",
        label="GP",
    )
    plt.scatter(
        [0.0898, -0.0898],
        [-0.7126, 0.7126],
        color="yellow",
        marker="*",
        s=150,
        label="Global Minima",
    )
    plt.title("GP Evaluations")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    # STP (nu=5)
    plt.subplot(1, 3, 2)
    contour = plt.contourf(X1, X2, Z, 50, cmap="viridis", alpha=0.7)
    plt.scatter(
        results_stp5["X"][:20, 0],
        results_stp5["X"][:20, 1],
        c="white",
        marker="o",
        label="Initial",
    )
    plt.scatter(
        results_stp5["X"][20:, 0],
        results_stp5["X"][20:, 1],
        c="green",
        marker="x",
        label="STP (nu=5)",
    )
    plt.scatter(
        [0.0898, -0.0898],
        [-0.7126, 0.7126],
        color="yellow",
        marker="*",
        s=150,
        label="Global Minima",
    )
    plt.title("STP (nu=5) Evaluations")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    # STP (nu=11)
    plt.subplot(1, 3, 3)
    contour = plt.contourf(X1, X2, Z, 50, cmap="viridis", alpha=0.7)
    plt.scatter(
        results_stp11["X"][:20, 0],
        results_stp11["X"][:20, 1],
        c="white",
        marker="o",
        label="Initial",
    )
    plt.scatter(
        results_stp11["X"][20:, 0],
        results_stp11["X"][20:, 1],
        c="blue",
        marker="x",
        label="STP (nu=11)",
    )
    plt.scatter(
        [0.0898, -0.0898],
        [-0.7126, 0.7126],
        color="yellow",
        marker="*",
        s=150,
        label="Global Minima",
    )
    plt.title("STP (nu=11) Evaluations")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    plt.tight_layout()
    plt.savefig("evaluations_comparison.png")
    plt.close()


def compare_processes_multiple_runs(n_runs=100):
    """Compare GP and STP on multiple runs as in the paper"""

    # Define the function to optimize
    def func(x1, x2):
        return six_hump_camel(x1, x2)

    # Define bounds
    bounds = [(-3, 3), (-2, 2)]

    # Initialize results storage
    best_y_history_gp = []
    best_y_history_stp5 = []
    best_y_history_stp11 = []

    for run in trange(n_runs):
        print(f"Run {run+1}/{n_runs}")

        # Generate initial samples (same for all models)
        np.random.seed(run)
        X_init = latin_hypercube_sampling(20, bounds)
        y_init = np.array([func(*x) for x in X_init])

        # Run GP
        results_gp = run_bayesian_optimization(
            func, bounds, n_init=20, n_iter=100, model_type="gp"
        )

        # Run STP (nu=5)
        results_stp5 = run_bayesian_optimization(
            func, bounds, n_init=20, n_iter=100, model_type="stp", nu=5
        )

        # Run STP (nu=11)
        results_stp11 = run_bayesian_optimization(
            func, bounds, n_init=20, n_iter=100, model_type="stp", nu=11
        )

        # Store results
        best_y_history_gp.append(results_gp["best_y_history"])
        best_y_history_stp5.append(results_stp5["best_y_history"])
        best_y_history_stp11.append(results_stp11["best_y_history"])

    # Convert to numpy arrays for easy manipulation
    # Make all histories the same length by padding with the last value
    max_len = max(
        max(len(hist) for hist in best_y_history_gp),
        max(len(hist) for hist in best_y_history_stp5),
        max(len(hist) for hist in best_y_history_stp11),
    )

    def pad_history(history, max_len):
        padded = np.zeros((len(history), max_len))
        for i, hist in enumerate(history):
            padded[i, : len(hist)] = hist
            if len(hist) < max_len:
                padded[i, len(hist) :] = hist[-1]
        return padded

    gp_hist = pad_history(best_y_history_gp, max_len)
    stp5_hist = pad_history(best_y_history_stp5, max_len)
    stp11_hist = pad_history(best_y_history_stp11, max_len)

    # Calculate log10 of error
    global_opt = -1.0316
    gp_err = np.log10(np.maximum(1e-10, np.abs(gp_hist - global_opt)))
    stp5_err = np.log10(np.maximum(1e-10, np.abs(stp5_hist - global_opt)))
    stp11_err = np.log10(np.maximum(1e-10, np.abs(stp11_hist - global_opt)))

    # Calculate median and quartiles
    gp_median = np.median(gp_err, axis=0)
    gp_q1 = np.percentile(gp_err, 25, axis=0)
    gp_q3 = np.percentile(gp_err, 75, axis=0)

    stp5_median = np.median(stp5_err, axis=0)
    stp5_q1 = np.percentile(stp5_err, 25, axis=0)
    stp5_q3 = np.percentile(stp5_err, 75, axis=0)

    stp11_median = np.median(stp11_err, axis=0)
    stp11_q1 = np.percentile(stp11_err, 25, axis=0)
    stp11_q3 = np.percentile(stp11_err, 75, axis=0)

    # Plot results similar to Figure 4 in the paper
    plt.figure(figsize=(10, 6))

    # x-axis is function evaluations: 20 initial + iterations
    x_vals = np.arange(max_len)

    plt.plot(x_vals, gp_median, "r-", label="GP")
    plt.fill_between(x_vals, gp_q1, gp_q3, color="r", alpha=0.2)

    plt.plot(x_vals, stp5_median, "b-", label="STP (nu=5)")
    plt.fill_between(x_vals, stp5_q1, stp5_q3, color="b", alpha=0.2)

    plt.plot(x_vals, stp11_median, "g-", label="STP (nu=11)")
    plt.fill_between(x_vals, stp11_q1, stp11_q3, color="g", alpha=0.2)

    plt.xlabel("Function evaluations")
    plt.ylabel("log10(|y - y*|)")
    plt.title("Comparison of GP vs. STP on Six-Hump Camel Function")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("comparison_multiple_runs.png")
    plt.close()

    # Print summary statistics
    final_errs_gp = gp_err[:, -1]
    final_errs_stp5 = stp5_err[:, -1]
    final_errs_stp11 = stp11_err[:, -1]

    print("\nSummary Statistics:")
    print(f"GP - Median final error: {10**np.median(final_errs_gp):.6f}")
    print(f"STP (nu=5) - Median final error: {10**np.median(final_errs_stp5):.6f}")
    print(f"STP (nu=11) - Median final error: {10**np.median(final_errs_stp11):.6f}")

    print(f"\nPercentage of runs finding global optimum (error < 1e-4):")
    print(f"GP: {np.mean(10**final_errs_gp < 1e-4) * 100:.1f}%")
    print(f"STP (nu=5): {np.mean(10**final_errs_stp5 < 1e-4) * 100:.1f}%")
    print(f"STP (nu=11): {np.mean(10**final_errs_stp11 < 1e-4) * 100:.1f}%")


def visualize_process_comparison():
    """Visualize comparison between GP and STP similar to Figures 1 and 2 in the paper"""
    # 1D domain for visualization
    x = np.linspace(0, 1, 100).reshape(-1, 1)

    # Set random seed
    np.random.seed(42)

    # Generate prior samples from GP and STP (nu=5)
    gp = GaussianProcess(bandwidth=0.1)
    stp = StudentTProcess(nu=3, bandwidth=0.1)

    # Function to generate samples from a process with numerical stability fix
    def generate_samples(process, x, n_samples=300):
        samples = []
        for _ in range(n_samples):
            # For GP, we can sample directly
            if isinstance(process, GaussianProcess):
                mu = np.zeros(len(x))
                K = process.compute_kernel_matrix(x)
                # Add small jitter to diagonal for numerical stability
                K = K + 1e-8 * np.eye(len(x))
                sample = np.random.multivariate_normal(mu, K)
            # For STP, we need to use the multivariate t distribution
            else:
                mu = np.zeros(len(x))
                Sigma = process.compute_kernel_matrix(x)
                # Add small jitter to diagonal for numerical stability
                Sigma = Sigma + 1e-8 * np.eye(len(x))
                # Generate from multivariate t using normal-wishart trick
                df = process.nu
                chol = np.linalg.cholesky(Sigma)
                z = np.random.normal(size=len(x))
                u = np.random.chisquare(df)
                sample = mu + chol @ z * np.sqrt(df / u)
            samples.append(sample)
        return np.array(samples)

    # Generate prior samples
    gp_samples = generate_samples(gp, x)
    stp_samples = generate_samples(stp, x)

    # Plot prior samples comparison (Figure 1)
    plt.figure(figsize=(14, 6))

    # GP prior
    plt.subplot(1, 2, 1)
    plt.plot(x, gp_samples.T, "gray", alpha=0.1)
    plt.plot(x, np.mean(gp_samples, axis=0), "darkgreen", linewidth=2)
    std = np.std(gp_samples, axis=0)
    plt.plot(x, np.mean(gp_samples, axis=0) + 2 * std, "green", linewidth=2)
    plt.plot(x, np.mean(gp_samples, axis=0) - 2 * std, "green", linewidth=2)
    plt.title("Gaussian Process Prior")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim(-3, 3)

    # STP prior
    plt.subplot(1, 2, 2)
    plt.plot(x, stp_samples.T, "gray", alpha=0.1)
    plt.plot(x, np.mean(stp_samples, axis=0), "darkgreen", linewidth=2)
    std = np.std(stp_samples, axis=0)
    plt.plot(x, np.mean(stp_samples, axis=0) + 2 * std, "green", linewidth=2)
    plt.plot(x, np.mean(stp_samples, axis=0) - 2 * std, "green", linewidth=2)
    plt.title("Student's-T Process Prior (nu=5)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim(-3, 3)

    plt.tight_layout()
    plt.savefig("prior_comparison.png")
    plt.close()

    # Generate observations for posterior visualization
    x_obs = np.array([0.2, 0.5, 0.8, 0.4, 0.3, 0.1, 0.6, 0.35]).reshape(-1, 1)
    y_obs = np.sin(2 * np.pi * x_obs).flatten()
    y_obs[4] += 1.0  # Add noise to one observation
    y_obs[-1] -= 1.0  # Add noise to one observation

    # Fit the processes
    gp.fit(x_obs, y_obs)
    stp.fit(x_obs, y_obs)

    # Generate posterior samples
    def generate_posterior_samples(process, x, x_obs, y_obs, n_samples=300):
        samples = []
        mean, std = process.predict(x, return_std=True)

        # For GP
        if isinstance(process, GaussianProcess):
            for _ in range(n_samples):
                sample = np.random.normal(mean, std)
                samples.append(sample)
        # For STP
        else:
            for _ in range(n_samples):
                # Generate from t distribution
                df = process.nu + len(x_obs)
                sample = mean + std * np.random.standard_t(df, size=len(x))
                samples.append(sample)

        return np.array(samples)

    # Generate posterior samples
    gp_post_samples = generate_posterior_samples(gp, x, x_obs, y_obs)
    stp_post_samples = generate_posterior_samples(stp, x, x_obs, y_obs)

    # Plot posterior comparison (Figure 2)
    plt.figure(figsize=(14, 6))

    # GP posterior
    plt.subplot(1, 2, 1)
    plt.plot(x, gp_post_samples.T, "gray", alpha=0.1)
    gp_mean, gp_std = gp.predict(x, return_std=True)
    plt.plot(x, gp_mean, "darkgreen", linewidth=2)
    plt.plot(x, gp_mean + 2 * gp_std, "green", linewidth=2)
    plt.plot(x, gp_mean - 2 * gp_std, "green", linewidth=2)
    plt.scatter(x_obs, y_obs, color="black", s=50)
    plt.title("Gaussian Process Posterior")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim(-3, 3)

    # STP posterior
    plt.subplot(1, 2, 2)
    plt.plot(x, stp_post_samples.T, "gray", alpha=0.1)
    stp_mean, stp_std = stp.predict(x, return_std=True)
    plt.plot(x, stp_mean, "darkgreen", linewidth=2)
    plt.plot(x, stp_mean + 2 * stp_std, "green", linewidth=2)
    plt.plot(x, stp_mean - 2 * stp_std, "green", linewidth=2)
    plt.scatter(x_obs, y_obs, color="black", s=50)
    plt.title("Student's-T Process Posterior (nu=5)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim(-3, 3)

    plt.tight_layout()
    plt.savefig("posterior_comparison.png")
    plt.close()


if __name__ == "__main__":
    # Test the six-hump camel function
    test_six_hump_camel_function()

    # Visualize process comparison
    print("Visualizing process comparison...")
    visualize_process_comparison()

    # # Compare processes on a single run
    # print("Comparing processes on a single run...")
    # compare_processes_single_run()

    # # Compare processes on multiple runs (reduced to 10 for speed)
    # print("Comparing processes on multiple runs...")
    # compare_processes_multiple_runs(
    #     n_runs=10
    # )  # Use 100 for full comparison as in paper
