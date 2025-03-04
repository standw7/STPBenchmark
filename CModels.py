import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from scipy.special import gamma
from scipy.optimize import minimize
from matplotlib.colors import LogNorm
from tqdm import trange


class StudentTProcess:
    def __init__(self, nu=5, kernel="squared_exp", bandwidth=1.0, mean_func=None):
        """
        Initialize a Student-T Process

        Parameters:
        -----------
        nu : float
            Degrees of freedom parameter, should be > 2
        kernel : str
            Kernel function to use ('squared_exp' for isotropic squared exponential)
        bandwidth : float
            Bandwidth parameter for the kernel
        mean_func : callable or None
            Mean function, if None, zero mean is used
        """
        self.nu = nu
        self.kernel_type = kernel
        self.bandwidth = bandwidth
        self.mean_func = mean_func if mean_func is not None else lambda x: 0

        # Data storage
        self.X_train = None
        self.y_train = None
        self.K_train_inv = None

    def kernel(self, x1, x2):
        """
        Compute the kernel function between two inputs

        Parameters:
        -----------
        x1, x2 : array-like
            Input vectors

        Returns:
        --------
        k : float
            Kernel value
        """
        if self.kernel_type == "squared_exp":
            # Isotropic squared exponential kernel
            return np.exp(-0.5 * np.sum((x1 - x2) ** 2) / (self.bandwidth**2))
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def compute_kernel_matrix(self, X1, X2=None):
        """
        Compute the kernel matrix for two sets of inputs with added jitter

        Parameters:
        -----------
        X1 : array-like, shape=(n_samples_1, n_features)
            First set of input vectors
        X2 : array-like, shape=(n_samples_2, n_features), optional
            Second set of input vectors, if None, X1 is used

        Returns:
        --------
        K : ndarray, shape=(n_samples_1, n_samples_2) or (n_samples_1, n_samples_1)
            Kernel matrix
        """
        X1 = np.atleast_2d(X1)

        if X2 is None:
            X2 = X1
            add_jitter = True  # Only add jitter to diagonal when X1 == X2
        else:
            X2 = np.atleast_2d(X2)
            add_jitter = False

        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel(X1[i], X2[j])

        # Add small jitter to diagonal for numerical stability
        if add_jitter:
            K = K + 1e-8 * np.eye(n1)

        return K

    def fit(self, X, y):
        """
        Fit the Student-T Process with training data

        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            Training input vectors
        y : array-like, shape=(n_samples,)
            Target values
        """
        self.X_train = np.atleast_2d(X)
        self.y_train = np.array(y).flatten()

        # Compute kernel matrix
        K = self.compute_kernel_matrix(self.X_train)

        # Compute inverse of kernel matrix
        self.K_train_inv = np.linalg.inv(K)

    def predict(self, X_test, return_std=False):
        """
        Predict using the Student-T Process

        Parameters:
        -----------
        X_test : array-like, shape=(n_samples, n_features)
            Test input vectors
        return_std : bool
            If True, also return standard deviation

        Returns:
        --------
        y_pred : ndarray, shape=(n_samples,)
            Predicted mean
        y_std : ndarray, shape=(n_samples,), optional
            Predicted standard deviation (only if return_std=True)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fit yet.")

        X_test = np.atleast_2d(X_test)

        # Compute kernel matrices
        K_star = self.compute_kernel_matrix(X_test, self.X_train)
        K_star_star = self.compute_kernel_matrix(X_test)

        # Compute posterior mean (equation 15)
        mu = K_star @ self.K_train_inv @ self.y_train

        if not return_std:
            return mu

        # Compute posterior covariance (equation 16)
        y_train_factor = self.y_train @ self.K_train_inv @ self.y_train
        n_train = len(self.y_train)
        cov_scale = (self.nu + y_train_factor - 2) / (self.nu + n_train - 2)

        cov = cov_scale * (K_star_star - K_star @ self.K_train_inv @ K_star.T)
        std = np.sqrt(np.diag(cov))

        return mu, std

    def expected_improvement(self, X, y_best):
        """
        Compute the expected improvement for new points

        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            Points to evaluate
        y_best : float
            Current best objective value

        Returns:
        --------
        ei : ndarray, shape=(n_samples,)
            Expected improvement for each point
        """
        X = np.atleast_2d(X)

        # Predict mean and std
        mu, std = self.predict(X, return_std=True)

        # Standardized value
        z = (y_best - mu) / std

        # Compute EI using equation 27
        pdf = student_t.pdf(x=z, df=self.nu + len(self.y_train))
        cdf = student_t.cdf(x=z, df=self.nu + len(self.y_train))

        t_factor = (self.nu + len(self.y_train) - 1) / (self.nu + len(self.y_train) - 2)
        t_factor *= 1 + z**2 / (self.nu + len(self.y_train))

        ei = (y_best - mu) * cdf + std * t_factor * pdf

        return ei

    def log_marginal_likelihood(self):
        """
        Compute the log marginal likelihood of the data

        Returns:
        --------
        lml : float
            Log marginal likelihood
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fit yet.")

        n_samples = len(self.y_train)
        K = self.compute_kernel_matrix(self.X_train)

        # Equation 28, simplified for zero mean
        lml = (
            gamma((self.nu + n_samples) / 2) / gamma(self.nu / 2)
            - n_samples / 2 * np.log(self.nu * np.pi)
            - 0.5 * np.log(np.linalg.det(K))
            - (self.nu + n_samples)
            / 2
            * np.log(1 + self.y_train @ self.K_train_inv @ self.y_train / self.nu)
        )

        return lml

    def optimize_hyperparams(self, param_grid=None):
        """
        Optimize kernel hyperparameters using grid search

        Parameters:
        -----------
        param_grid : dict or None
            Dictionary with parameter names as keys and lists of values to try
            If None, a default grid for bandwidth is used

        Returns:
        --------
        best_params : dict
            Best parameters
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fit yet.")

        if param_grid is None:
            # Default grid for bandwidth
            log_bandwidths = np.linspace(-3, 3, 11)
            param_grid = {"bandwidth": np.exp(log_bandwidths)}

        best_lml = -np.inf
        best_params = {}

        for param_name, param_values in param_grid.items():
            for value in param_values:
                # Set parameter
                setattr(self, param_name, value)

                # Recompute kernel matrix and its inverse
                K = self.compute_kernel_matrix(self.X_train)
                self.K_train_inv = np.linalg.inv(K)

                # Compute log marginal likelihood
                lml = self.log_marginal_likelihood()

                if lml > best_lml:
                    best_lml = lml
                    best_params[param_name] = value

        # Set the best parameters
        for param_name, value in best_params.items():
            setattr(self, param_name, value)

        # Recompute kernel matrix and its inverse with best parameters
        K = self.compute_kernel_matrix(self.X_train)
        self.K_train_inv = np.linalg.inv(K)

        return best_params


class GaussianProcess:
    def __init__(self, kernel="squared_exp", bandwidth=1.0, mean_func=None):
        """
        Initialize a Gaussian Process (simplified version for comparison)

        Parameters:
        -----------
        kernel : str
            Kernel function to use ('squared_exp' for isotropic squared exponential)
        bandwidth : float
            Bandwidth parameter for the kernel
        mean_func : callable or None
            Mean function, if None, zero mean is used
        """
        self.kernel_type = kernel
        self.bandwidth = bandwidth
        self.mean_func = mean_func if mean_func is not None else lambda x: 0

        # Data storage
        self.X_train = None
        self.y_train = None
        self.K_train_inv = None

    def kernel(self, x1, x2):
        """Same as StudentTProcess.kernel"""
        if self.kernel_type == "squared_exp":
            return np.exp(-0.5 * np.sum((x1 - x2) ** 2) / (self.bandwidth**2))
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def compute_kernel_matrix(self, X1, X2=None):
        """
        Compute the kernel matrix for two sets of inputs with added jitter

        Parameters:
        -----------
        X1 : array-like, shape=(n_samples_1, n_features)
            First set of input vectors
        X2 : array-like, shape=(n_samples_2, n_features), optional
            Second set of input vectors, if None, X1 is used

        Returns:
        --------
        K : ndarray, shape=(n_samples_1, n_samples_2) or (n_samples_1, n_samples_1)
            Kernel matrix
        """
        X1 = np.atleast_2d(X1)

        if X2 is None:
            X2 = X1
            add_jitter = True  # Only add jitter to diagonal when X1 == X2
        else:
            X2 = np.atleast_2d(X2)
            add_jitter = False

        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel(X1[i], X2[j])

        # Add small jitter to diagonal for numerical stability
        if add_jitter:
            K = K + 1e-8 * np.eye(n1)

        return K

    def fit(self, X, y):
        """Same as StudentTProcess.fit"""
        self.X_train = np.atleast_2d(X)
        self.y_train = np.array(y).flatten()

        # Compute kernel matrix
        K = self.compute_kernel_matrix(self.X_train)

        # Compute inverse of kernel matrix
        self.K_train_inv = np.linalg.inv(K)

    def predict(self, X_test, return_std=False):
        """
        Predict using the Gaussian Process (key difference from STP)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fit yet.")

        X_test = np.atleast_2d(X_test)

        # Compute kernel matrices
        K_star = self.compute_kernel_matrix(X_test, self.X_train)
        K_star_star = self.compute_kernel_matrix(X_test)

        # Compute posterior mean (same as STP, equation 6)
        mu = K_star @ self.K_train_inv @ self.y_train

        if not return_std:
            return mu

        # Compute posterior covariance (equation 7, different from STP)
        cov = K_star_star - K_star @ self.K_train_inv @ K_star.T
        std = np.sqrt(np.diag(cov))

        return mu, std

    def expected_improvement(self, X, y_best):
        """
        Compute the expected improvement for new points (Gaussian version)
        """
        X = np.atleast_2d(X)

        # Predict mean and std
        mu, std = self.predict(X, return_std=True)

        # Standardized value
        z = (y_best - mu) / std

        # Compute EI using equation 9
        from scipy.stats import norm

        pdf = norm.pdf(z)
        cdf = norm.cdf(z)

        ei = (y_best - mu) * cdf + std * pdf

        return ei

    def log_marginal_likelihood(self):
        """
        Compute the log marginal likelihood of the data (Gaussian version)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fit yet.")

        n_samples = len(self.y_train)
        K = self.compute_kernel_matrix(self.X_train)

        # Standard Gaussian log marginal likelihood
        lml = (
            -0.5 * self.y_train @ self.K_train_inv @ self.y_train
            - 0.5 * np.log(np.linalg.det(K))
            - n_samples / 2 * np.log(2 * np.pi)
        )

        return lml

    def optimize_hyperparams(self, param_grid=None):
        """Same as StudentTProcess.optimize_hyperparams"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fit yet.")

        if param_grid is None:
            # Default grid for bandwidth
            log_bandwidths = np.linspace(-3, 3, 11)
            param_grid = {"bandwidth": np.exp(log_bandwidths)}

        best_lml = -np.inf
        best_params = {}

        for param_name, param_values in param_grid.items():
            for value in param_values:
                # Set parameter
                setattr(self, param_name, value)

                # Recompute kernel matrix and its inverse
                K = self.compute_kernel_matrix(self.X_train)
                self.K_train_inv = np.linalg.inv(K)

                # Compute log marginal likelihood
                lml = self.log_marginal_likelihood()

                if lml > best_lml:
                    best_lml = lml
                    best_params[param_name] = value

        # Set the best parameters
        for param_name, value in best_params.items():
            setattr(self, param_name, value)

        # Recompute kernel matrix and its inverse with best parameters
        K = self.compute_kernel_matrix(self.X_train)
        self.K_train_inv = np.linalg.inv(K)

        return best_params


def six_hump_camel(x1, x2):
    """
    Six-hump camel function

    Parameters:
    -----------
    x1 : float
        First input, in [-3, 3]
    x2 : float
        Second input, in [-2, 2]

    Returns:
    --------
    f : float
        Function value
    """
    term1 = (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2

    return term1 + term2 + term3


def latin_hypercube_sampling(n_samples, bounds):
    """
    Generate samples using Latin Hypercube Sampling

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    bounds : array-like, shape=(n_dims, 2)
        Min and max values for each dimension

    Returns:
    --------
    samples : ndarray, shape=(n_samples, n_dims)
        Generated samples
    """
    n_dims = len(bounds)

    # Generate samples in [0, 1]^n_dims
    samples = np.zeros((n_samples, n_dims))

    for j in range(n_dims):
        perms = np.random.permutation(n_samples)
        samples[:, j] = (perms + np.random.random(n_samples)) / n_samples

    # Scale to actual bounds
    for j in range(n_dims):
        samples[:, j] = bounds[j][0] + samples[:, j] * (bounds[j][1] - bounds[j][0])

    return samples


def normalize_data(X, y):
    """
    Normalize input and output data to have zero mean and unit variance

    Parameters:
    -----------
    X : array-like, shape=(n_samples, n_features)
        Input data
    y : array-like, shape=(n_samples,)
        Output data

    Returns:
    --------
    X_norm : ndarray, shape=(n_samples, n_features)
        Normalized input data
    y_norm : ndarray, shape=(n_samples,)
        Normalized output data
    X_mean : ndarray, shape=(n_features,)
        Mean of input data
    X_std : ndarray, shape=(n_features,)
        Standard deviation of input data
    y_mean : float
        Mean of output data
    y_std : float
        Standard deviation of output data
    """
    X = np.atleast_2d(X)
    y = np.array(y).flatten()

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    y_mean = np.mean(y)
    y_std = np.std(y)

    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std

    return X_norm, y_norm, X_mean, X_std, y_mean, y_std


def run_bayesian_optimization(
    func, bounds, n_init=20, n_iter=100, model_type="gp", nu=5, n_restarts=100
):
    """
    Run Bayesian optimization

    Parameters:
    -----------
    func : callable
        Function to optimize
    bounds : array-like, shape=(n_dims, 2)
        Min and max values for each dimension
    n_init : int
        Number of initial samples
    n_iter : int
        Number of iterations
    model_type : str
        Type of model to use ('gp' for Gaussian Process, 'stp' for Student's-T Process)
    nu : float
        Degrees of freedom parameter for STP
    n_restarts : int
        Number of optimization restarts

    Returns:
    --------
    results : dict
        Dictionary containing optimization results
    """
    n_dims = len(bounds)

    # Generate initial samples
    X_init = latin_hypercube_sampling(n_init, bounds)
    y_init = np.array([func(*x) for x in X_init])

    # Initialize best values
    best_x = X_init[np.argmin(y_init)]
    best_y = np.min(y_init)

    # Initialize results tracking
    X_all = X_init.copy()
    y_all = y_init.copy()
    best_y_history = [best_y]

    # Initialize model
    if model_type.lower() == "gp":
        model = GaussianProcess()
    elif model_type.lower() == "stp":
        model = StudentTProcess(nu=nu)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Re-normalize data every 10 iterations
    renorm_interval = 10

    for i in trange(n_iter):
        # Re-normalize data
        if i % renorm_interval == 0:
            X_norm, y_norm, X_mean, X_std, y_mean, y_std = normalize_data(X_all, y_all)

            # Fit model
            model.fit(X_norm, y_norm)

            # Optimize hyperparameters
            # Two-step grid search as described in the paper
            log_bandwidths1 = np.linspace(-3, 3, 11)
            param_grid1 = {"bandwidth": np.exp(log_bandwidths1)}
            best_params1 = model.optimize_hyperparams(param_grid1)

            center = np.log(best_params1["bandwidth"])
            log_bandwidths2 = np.linspace(center - 0.5, center + 0.5, 11)
            param_grid2 = {"bandwidth": np.exp(log_bandwidths2)}
            model.optimize_hyperparams(param_grid2)

        # Find next point to evaluate using expected improvement
        best_y_norm = (best_y - y_mean) / y_std

        # First, search on a grid
        grid_size = 101
        x1_grid = np.linspace(bounds[0][0], bounds[0][1], grid_size)
        x2_grid = np.linspace(bounds[1][0], bounds[1][1], grid_size)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

        # Normalize grid points
        X_grid_norm = (X_grid - X_mean) / X_std

        # Compute EI for all grid points
        ei_grid = model.expected_improvement(X_grid_norm, best_y_norm)

        # Find the point with highest EI
        best_idx = np.argmax(ei_grid)
        x_next_init = X_grid[best_idx]

        # Use local optimization to refine
        def negative_ei(x):
            x_norm = (x - X_mean) / X_std
            return -model.expected_improvement(x_norm.reshape(1, -1), best_y_norm)[0]

        res = minimize(negative_ei, x_next_init, method="L-BFGS-B", bounds=bounds)
        x_next = res.x

        # Evaluate function at the new point
        y_next = func(*x_next)

        # Update dataset
        X_all = np.vstack([X_all, x_next])
        y_all = np.append(y_all, y_next)

        # Update best value
        if y_next < best_y:
            best_y = y_next
            best_x = x_next

        best_y_history.append(best_y)

        # Check for convergence (found the global optimum)
        global_opt = -1.0316
        if abs(best_y - global_opt) < 1e-4:
            break

    results = {
        "X": X_all,
        "y": y_all,
        "best_x": best_x,
        "best_y": best_y,
        "best_y_history": best_y_history,
        "n_evaluations": len(y_all),
    }

    return results
