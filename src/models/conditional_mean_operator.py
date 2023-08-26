"""This module implements an estimator for the conditional expectation
operator from L2(X) to L2(Z). The implementation is based on the first stage of
the Kernel Instrumental Variable algorithm, and relies on kernel methods.

Author: @Caioflp

"""
import numpy as np

from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator


class ConditionalMeanOperator(BaseEstimator):
    def __init__(
        self,
        regularization_weight: float
    ):
        self.regularization_weight = regularization_weight
        self.lengthscale = None
        self.kernel_gramian_regularized = None
        self.loop_weights = None
        self.n_samples = None
        self.loop_fitted = None
        self.z_samples = None


    def kernel(self, z_1: np.ndarray, z_2: np.ndarray):
        """Kernel used for fiting.

        Computes the gramiam matrix of the kernel with respect to both vectors.

        Parameters
        ----------
        z_1: np.ndarray
            Array with shape (n_1, dim)
        z_2: np.ndarray
            Array with shape (n_2, dim)

        Returns
        -------
            Array with shape (n_1, n_2)

        """
        assert len(z_1.shape) == len(z_2.shape) == 2
        squared_distances = distance_matrix(z_1, z_2)**2
        return np.exp(- self.lengthscale * squared_distances)

    def loop_fit(self, z_train: np.ndarray, z_loop: np.ndarray):
        assert len(z_train.shape) == len(z_loop.shape) == 2
        assert z_train.shape[1] == z_loop.shape[1]
        self.n_samples = z_train.shape[0]

        median = np.median(
            np.ravel(distance_matrix(z_train, z_train))
        )
        self.lengthscale = 1 / median

        self.loop_weights = np.linalg.solve(
            (
                self.kernel(z_train, z_train)
                + self.regularization_weight*np.eye(n_samples)
            ),
            self.kernel(z_train, z_loop)

        )
        self.loop_fitted = True

    def fit(self, z_samples: np.ndarray):
        self.z_samples = z_samples
        self.n_samples = z_samples.shape[0]

        median = np.median(
            np.ravel(distance_matrix(z_samples, z_samples))
        )
        self.lengthscale = 1 / median

        self.kernel_gramian_regularized = (
            self.kernel(z_samples, z_samples)
            + self.regularization_weight*np.eye(n_samples)
        )

    def loop_predict(self, f_samples: np.ndarray, it: int):
        """Predict method for use within the loop in the main model.
        `it` is the iteration number, starting from 0.

        f_samples must be one dimensional

        """
        assert len(f_samples.shape) == 1
        assert f_samples.size == self.n_samples
        return self.loop_weights[:, it] @ f_samples

    def predict(self, f_samples: np.ndarray, z: np.ndarray):
        assert len(z.shape) == 1
        assert len(f_samples.shape) == 1
        assert f_samples.size == self.n_samples
        weights = np.linalg.solve(
            self.kernel_gramian_regularized,
            self.kernel(self.z_samples, z.reshape(1, -1))
        )
        return weights @ f_samples
