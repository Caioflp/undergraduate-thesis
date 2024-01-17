"""Model related utilities.

Author: @Caioflp

"""
import abc
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neighbors import KernelDensity as KDE


class Domain:
    def __init__(
        self,
        observed_points: np.ndarray,
        grid_points: np.ndarray
    ) -> None:
        assert len(observed_points.shape) == len(grid_points.shape) == 2
        assert observed_points.shape[1] == grid_points.shape[1]
        self.spliting_index = observed_points.shape[0]
        self._all_points = np.concatenate(
            (observed_points, grid_points),
            axis=0
        )

    @property
    def all_points(self):
        return self._all_points

    @property
    def observed_points(self):
        return self._all_points[:self.spliting_index]

    @property
    def grid_points(self):
        return self._all_points[self.spliting_index:]


class Estimates:
    """Holds sequence of estimates evaluated on grid points and on observed
    points.

    """
    def __init__(
        self,
        n_estimates: int,
        n_observed_points: int,
        n_grid_points: int,
    ) -> None:
        self.n_estimates = n_estimates
        self.n_observed_points = n_observed_points
        self.n_grid_points = n_grid_points

        # Observed points come first!
        self.spliting_index = self.n_observed_points
        self._estimates = np.empty(
            (n_estimates, n_observed_points + n_grid_points),
            dtype=np.float64
        )

    @property
    def on_all_points(self) -> np.ndarray:
        return self._estimates

    @property
    def on_observed_points(self) -> np.ndarray:
        return self._estimates[:, :self.spliting_index]

    @property
    def on_grid_points(self) -> np.ndarray:
        return self._estimates[:, self.spliting_index:]


class FinalEstimate:
    def __init__(
        self,
        on_observed_points: np.ndarray,
        on_grid_points: np.ndarray,
    ) -> None:
        assert len(on_observed_points.shape) == len(on_grid_points.shape) == 1
        self.spliting_index = on_observed_points.shape[0]
        self._estimate = np.concatenate(
            (on_observed_points, on_grid_points),
        )

    @property
    def on_observed_points(self):
        return self._estimate[:self.spliting_index]
    
    @property
    def on_grid_points(self):
        return self._estimate[self.spliting_index:]
    
    @property
    def on_all_points(self):
        return self._estimate


class Loss(abc.ABC):
    """ Base class for pointwise loss function.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(
        self,
        y: np.ndarray,
        y_prime: np.ndarray,
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def derivative_second_argument(
        self,
        y: np.ndarray,
        y_prime: np.ndarray,
    ) -> np.ndarray:
        pass

class QuadraticLoss(Loss):
    """ Quadratic loss.

    .. math::
        \ell (y, y') = 1/2 * (y - y')^2
    """
    def __init__(self):
        super().__init__()
    
    def __call__(
        self,
        y: np.ndarray,
        y_prime: np.ndarray,
    ) -> np.ndarray:
        assert y.shape == y_prime.shape
        return 0.5 * (y - y_prime)**2

    def derivative_second_argument(
        self,
        y: np.ndarray,
        y_prime: np.ndarray,
    ) -> np.ndarray:
        assert y.shape == y_prime.shape
        return y_prime - y


class BCELogisticLoss(Loss):
    """ BCE + logistic function loss

    .. math::
        \ell (y, y') = BCE(y, 1 - logistic(-y'))
    """
    def __init__(self):
        super().__init__()

    def logistic(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def BCE(
        self,
        p: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        return - ( p * np.log(q) + (1 - p) * np.log(q) )
    
    def __call__(
        self,
        y: np.ndarray,
        y_prime: np.ndarray,
    ) -> np.ndarray:
        assert y.shape == y_prime.shape
        return self.BCE(y, 1 - self.logistic(-y_prime))

    def derivative_second_argument(
        self,
        y: np.ndarray,
        y_prime: np.ndarray,
    ) -> np.ndarray:
        assert y.shape == y_prime.shape
        return self.logistic(y_prime) - y


def create_covering_grid(
    X: np.ndarray,
    step: float = 1E-2
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
        array of shape (n_samples,) or (n_samples, n_features)

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)

    Raises
    ------
    ValueError if the array has more than two dimensions.

    """
    if len(arr.shape) == 1:
        return arr.reshape(-1, 1)
    elif len(arr.shape) == 2:
        return arr
    else:
        raise ValueError


def truncate(arr: np.ndarray, M: float):
    """Truncate the values of `arr` inside [-M, M]
    """
    return np.minimum(np.maximum(arr, 0), M) - np.minimum(np.maximum(-arr, 0), M)


def distance_squared_matrix(X: np.ndarray, Y: np.ndarray):
    assert len(X.shape) == len(Y.shape) == 2
    assert X.shape[1] == Y.shape[1]
    X_i_squared = np.sum(X**2, axis=1, keepdims=True)
    Y_j_squared = np.sum(Y**2, axis=1, keepdims=True).T
    X_i_dot_Y_j = X@Y.T
    return (X_i_squared + Y_j_squared) - 2*X_i_dot_Y_j



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng()
    sample = rng.normal(size=(100, 2))
    domain = create_covering_grid(sample, step=1E-1)
    assert (domain.min(axis=0) == sample.min(axis=0)).all()
    assert (domain.max(axis=0) == sample.max(axis=0)).all()

    plt.scatter(domain[:, 0], domain[:, 1], c="b", s=.2)
    plt.scatter(sample[:, 0], sample[:, 1], c="r", s=5)
    plt.show()
