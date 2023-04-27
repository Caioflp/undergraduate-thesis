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
        lr: Literal["inv_sqrt", "inv_n_samples"],
    ):
        pass

    def fit(self, X: np.ndarray, Z: np.ndarray, Y: np.ndarray):
        """Fits the estimator to iid data.

        """

    def predict(self, X: np.ndarray):
        """Applies the fitted function to given inputs.

        """
