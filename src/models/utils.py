"""Implements utilities for the models to use.

Author: @Caioflp

"""
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


DEFAULT_REGRESSOR = KNeighborsRegressor(weights="distance", n_neighbors=5)


@dataclass
class Estimates:
    """Holds sequence of estimates evaluated on grid points and on observed
    points.

    Attributes
    ----------
    on_grid_points : np.ndarray
        Array of shape (n_estimates, n_grid_points) which contains the values
        of the estimate function on each of the grid points.
    on_observed_points : np.ndarray
        Array of shape (n_estimates, n_samples) which contains the values
        of the estimate function on each of the observed (sample) points

    """
    on_grid_points: np.ndarray
    on_observed_points: np.ndarray


def create_discretized_domain(
    X: np.ndarray,
    step: float = 1E-1
) -> np.ndarray:
    """Create a regular grid of points covering the domain of the input
    array X.

    Parameters
    ----------
    X : array_like
        Input array with shape (n_samples, n_features).
    step : float, optional
        The step size for the grid. Defaults to 1E-1.

    Returns
    -------
    ndarray
        A 2D array with shape (n_grid_points, n_features), where
        n_grid_points is the total number of points in the grid and
        n_features is the number of features in X.

    Notes
    -----
    This function creates a regular grid of points covering the domain of 
    the input array X. It treats the 1-dimensional case separately, using
    numpy.linspace to create the grid. For higher dimensions, it creates a
    meshgrid of intervals for each feature of X, then reshapes and stacks
    the resulting arrays to create the final grid.

    """
    if len(X.shape) == 1:
        X = X[:, None]
    dim = X.shape[1]
    lower_bound = X.min(axis=0)
    upper_bound = X.max(axis=0)
    amplitude = upper_bound - lower_bound
    # Treat the 1 dimensional case separately, as it is cheaper.
    if dim == 1:
        return np.linspace(
            lower_bound,
            upper_bound,
            int(amplitude/step),
        ).flatten()

    intervals = (
        np.linspace(lower_bound[i], upper_bound[i],
                    num=int(amplitude[i]/step))
        for i in range(dim)
    )
    separate_coordinates = np.meshgrid(*intervals)
    squished_separate_coordinates = [
        np.reshape(x, (-1, 1)) for x in separate_coordinates
    ]
    return np.hstack(squished_separate_coordinates)


def ensure_two_dimensional(arr: np.ndarray):
    """Ensures passed array has two dimensions
    
    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    np.ndarray

    Raises
    ------
    ValueError if the array has more than two dimensions

    """
    if len(arr.shape) == 1:
        return arr.reshape(-1, 1)
    elif len(arr.shape) == 2:
        return arr
    else:
        raise ValueError


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng()
    sample = rng.normal(size=(100, 2))
    domain = create_discretized_domain(sample, step=1E-1)
    assert (domain.min(axis=0) == sample.min(axis=0)).all()
    assert (domain.max(axis=0) == sample.max(axis=0)).all()

    plt.scatter(domain[:, 0], domain[:, 1], c="b", s=.2)
    plt.scatter(sample[:, 0], sample[:, 1], c="r", s=5)
    plt.show()
