"""Implementations of different models for IV regression.

Author: @Caioflp
"""
from typing import Literal

import numpy as np


class SGDIVProjectedLoss:
    """Implements a functional stochastic gradient descent algorithm for regression
    on IV problems.


    Notes
    -----
    In this formulation, the loss is computed after projecting onto Z, i.e.,
    .. math::
        R(h) = \mathbf{E} [\ell (r_0(Z), h(Z))],
    where :math:`r_0(Z) = \mathbf{E} [Y | Z]`.

    """
    def __init__(
        self,
        lr: Literal["inv_sqrt", "inv_n_samples"] = "inv_n_samples",
    ):
        pass

    def create_domain_grid(
        self,
        X: np.ndarray,
        step: float = 1E-1
    ) -> np.ndarray:
        """Creates a grid domain with the same dimensionality as X.

        This function returns an array of points which are the discretized
        version of the smallest hyperparallelepiped which contains the data in X.

        Parameters
        ----------
        X: np.ndarray
            A `(n, p)` dimensional array. In case `p` is 1, we also accept a
            `(n,)` dimensional array.

        Returns
        -------
        np.ndarray
            A `(N, p)` array, where N is the number of points of the
            discretization.

        """
        if len(X.shape) == 1:
            X = X[:, None]
        # dimension
        dim = X.shape[1]
        lower_bound = X.min(axis=0)
        upper_bound = X.max(axis=0)
        amplitude = upper_bound - lower_bound
        # Treat the 1 dimensional case separately, as it is cheaper
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

    def predict(self, X: np.ndarray):
        """Applies the fitted function to given inputs.

        """

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    model = SGDIVProjectedLoss()

    rng = np.random.default_rng()
    sample = rng.normal(size=(100, 2))
    domain = model.create_domain_grid(sample, step=1E-1)
    assert (domain.min(axis=0) == sample.min(axis=0)).all()
    assert (domain.max(axis=0) == sample.max(axis=0)).all()

    plt.scatter(domain[:, 0], domain[:, 1], c="b", s=.2)
    plt.scatter(sample[:, 0], sample[:, 1], c="r", s=5)
    plt.show()
