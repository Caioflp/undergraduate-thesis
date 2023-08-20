"""Implements density ratio estimation using the uLSIF algorithm, described in
chapter 6 of the book 'Density Ratio Estimation in Machine Learning', by
Sugiyama et al. The basis functions are gaussian kernels.

author: @Caioflp

"""
import numpy as np

from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator


class DensityRatio(BaseEstimator):
    def __init__(
        self,
        lengthscale: float,
        regularization_weight: float,
    ):
        self.lenghtscale = lengthscale
        self.regularization_weight = regularization_weight
        self.dim = None
        self.theta = None
        self.support_points = None
        self.fitted = False

    def __call__(self, w):
        assert self.fitted
        assert len(w.shape) == 1
        assert w.shape[0] == self.dim

        w = w.reshape(1, -1)
        return np.sum(self.theta * self.kernel(self.support_points, w))


    def kernel(w_1: np.ndarray, w_2: np.ndarray):
        """Kernel used for fiting.

        Computes the gramiam matrix of the kernel with respect to both vectors.

        Parameters
        ----------
        w_1: np.ndarray
            Array with shape (n_1, dim)
        w_2: np.ndarray
            Array with shape (n_2, dim)

        Returns
        -------
            Array with shape (n_1, n_2)

        """
        assert len(w_1.shape) == len(w_2.shape) == 2
        squared_distances = distance_matrix(w_1, w_2)**2
        return np.exp(- self.lenghscale * squared_distances)


    def fit(
        numerator_samples: np.ndarray,
        denominator_samples: np.ndarray,
        regularization: str
    ):
        """Fits estimator to provided samples.

        Uses the uLSIF algorithm, with gaussian kernel, to estimate the ratio
        of the densities of the distributions from the numerator and the
        denominator.
        Assumes that an equal number of samples from the numerator and the
        denominator was given.

        Parameters
        ----------
        numerator_samples: np.ndarray
            Array with shape `(n_samples, dim)`.
        denominator_samples: np.ndarray
            Array with shape `(n_samples, dim)`.
        regularization: string in ["l2", "rkhs"]
            Whether to use l2 norm of `theta` as regularizer or the RKHS norm
            of the estimator.

        """
        msg_n_samples = (
            "Provide the same number of samples for numerator and " +
            "denominator."
        )
        condition_n_samples = (
            numerator_samples.shape[0] == denominator_samples.shape[0]
        )
        assert condition_n_samples, msg_n_samples

        msg_dim = (
            "The dimension of numerator samples and denominator samples " +
            "must match."
        )
        condition_dim = numerator_samples.shape[1] == denominator_samples.shape[1]
        assert condition_dim, msg_dim

        assert regularization in ["l2", "rkhs"], "Unknown regularization"

        n_samples, self.dim = numerator_samples.shape
        self.support_points = numerator_samples

        K = self.kernel(numerator_samples, numerator_samples)
        h_hat = np.mean(K, axis=1, keepdims=True)
        cross_K = self.kernel(numerator_samples, denominator_samples)
        H_hat = cross_K @ cross_K.T / n_samples

        if regularization == "l2":
            self.theta = np.linalg.solve(
                H_hat + self.regularization_weight * np.eye(n_samples), h_hat
            )
        else:
            self.theta = np.linalg.solve(
                H_hat + self.regularization_weight * K, h_hat
            )
        self.fitted = True
