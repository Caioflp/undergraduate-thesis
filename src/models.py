"""Implementations of different models for IV regression.

Author: @Caioflp
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator


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


class FunctionalGradientDescentIV(BaseEstimator):
    """Implements a functional gradient descent algorithm for regression
    on IV problems.

    Parameters
    ----------
    lr : str, default "inv_n_samples"
        A string which determines which learning rate to use. `"inv_sqrt"`
        means `alpha_i = 1/sqrt(i)`. `"inv_n_samples"` means that the learning
        rate is constant and equal to the inverse square root of the number of
        training samples.
    projector_y : BaseEstimator, default KNN(n_neighbors=5)
        Algorithm used to project `Y` onto `Z`, i.e., to compute E [Y | Z = z]
        for the observed values of `z`.
    projector_estimate : BaseEstimator, default KNN(n_neighbors=5)
        Algorithm used to project `h(X)` onto `Z`, i.e., to compute
        E [h(X) | Z = z] for the observed values of `z`.
    regressor_grad : BaseEstimator, default KNN(n_neighbors=5)
        Algorithm used to regress the loss's gradient onto `Z`, i.e., to
        compute E [\partial_2 l (E[Y|Z], E[h(X)|Z]) | X = x], for the observed
        and grid values of `x`.

    Attributes
    ----------
    lr : str
        Type of learning rate for the fitting algorithm.
    projector_y : BaseEstimator
    projector_estimate : BaseEstimator
    projector_grad : BaseEstimator
    estimate_domain : np.ndarray
        An array of shape (n_samples + n_grid_points,) containing the points
        on which our estimate for the solution is being computed.
    estimate : np.ndarray
        An array of shape (n_samples + n_grid_points,) containing the values of
        our estimate computed on the points in `self.estimate_domain`.
    grid_domain : np.ndarray
        An array of shape (n_grid_points,) containg the values of
        `self.estimate_domain` which are grid points.
    estimate_on_grid : np.ndarray
        An array of shape (n_grid_points,) containing the the values of our
        estimate computed on the points in `self.grid_domain`.
    estimate_on_obs : np.ndarray
        An array of shape (n_samples,) containing the the values of our
        estimate computed on the points supplied as samples for the `fit`
        method.

    Notes
    -----
    In this formulation, the loss is computed after projecting onto Z, i.e.,
    .. math::
        R(h) = E [l (r_0(Z), Th(Z))],
    where :math:`r_0(Z) = E [Y | Z]` and 
    .. math::
        Th(Z) = E [h(X) | Z].

    """
    def __init__(
        self,
        lr: Literal["inv_sqrt", "inv_n_samples"] = "inv_n_samples",
        projector_y: BaseEstimator = DEFAULT_REGRESSOR,
        projector_estimate: BaseEstimator = DEFAULT_REGRESSOR,
        regressor_grad: BaseEstimator = DEFAULT_REGRESSOR,
    ):
        self.lr = lr
        self.projector_y = projector_y
        self.projector_estimate = projector_estimate
        self.regressor_grad = regressor_grad


    def create_discretized_domain(
        self,
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

    def fit(self, X: np.ndarray, Z: np.ndarray, Y: np.ndarray):
        """Fits the estimator to iid data.

        """
        # Take care of dimensions
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        n_iter = n_samples

        lr_dict = {
            "inv_n_samples": lambda i: 1/np.sqrt(n_samples),
            "inv_sqrt": lambda i: 1/np.sqrt(i)
        }
        lr = lr_dict[self.lr]

        # Create domain for estimates.
        x_domain = self.create_discretized_domain(X).reshape(-1, 1)
        n_grid_points = x_domain.shape[0]

        # Create object which will store the sequence of estimates evaluated on
        # unobserved grid points and on observed random points.
        estimates = Estimates(
            on_grid_points=np.empty((n_iter+1, n_grid_points), dtype=np.float64),
            on_observed_points=np.empty((n_iter+1, n_samples), dtype=np.float64)
        )
        estimates.on_grid_points[0, :] = np.zeros(n_grid_points)
        estimates.on_observed_points[0, :] = np.zeros(n_samples)

        # Compute the projected values of Y onto Z.
        projected_y = self.projector_y.fit(Z, Y).predict(Z)

        for i in range(1, n_iter+1):
            # Project current estimate on Z, i.e., compute E [Th(X) | Z]
            current_estimate = estimates.on_observed_points[i-1]
            projected_current_estimate = \
                    self.projector_estimate.fit(Z, current_estimate).predict(Z)

            # Compute gradient of loss function.
            # We are assuming that the loss function is quadratic, which makes
            # its gradient linear.
            # TODO: Implement other types of loss functions.
            loss_grad = projected_current_estimate - projected_y

            # Compute actual functional gradient, which is the regression of
            # the loss gradient on X, i.e.,
            # E [\partial_2 l (r_0(Z), Th(Z)) | X]
            self.regressor_grad.fit(X, loss_grad)
            functional_grad_grid = self.regressor_grad.predict(x_domain)
            functional_grad_observed = self.regressor_grad.predict(X)

            # Take one step in the negative gradient direction
            estimates.on_grid_points[i, :] =  (
                estimates.on_grid_points[i-1, :]
                - lr(i) * functional_grad_grid
            )
            estimates.on_observed_points[i, :] =  (
                estimates.on_observed_points[i-1, :]
                - lr(i) * functional_grad_observed
            )

        # Construct final estimate as average of sequence of estimates
        mean_estimate_on_grid = estimates.on_grid_points.mean(axis=0)
        mean_estimate_on_observed = estimates.on_observed_points.mean(axis=0)
        self.estimate_domain = np.concatenate((x_domain, X), axis=0)
        self.estimate = np.concatenate(
            (mean_estimate_on_grid, mean_estimate_on_observed),
            axis=None
        )
        self.grid_domain = x_domain
        self.estimate_on_grid = mean_estimate_on_grid
        self.estimate_on_obs = mean_estimate_on_observed


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = FunctionalGradientDescentIV()

    rng = np.random.default_rng()
    sample = rng.normal(size=(100, 2))
    domain = model.create_discretized_domain(sample, step=1E-1)
    assert (domain.min(axis=0) == sample.min(axis=0)).all()
    assert (domain.max(axis=0) == sample.max(axis=0)).all()

    plt.scatter(domain[:, 0], domain[:, 1], c="b", s=.2)
    plt.scatter(sample[:, 0], sample[:, 1], c="r", s=5)
    plt.show()
