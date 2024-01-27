"""Implements density ratio estimation using the uLSIF algorithm, described in
chapter 6 of the book 'Density Ratio Estimation in Machine Learning', by
Sugiyama et al. The basis functions are gaussian kernels.

author: @Caioflp

"""
import abc
import logging

import numpy as np

from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold


logger = logging.getLogger("src.models.density_ratio")


class DensityRatio(BaseEstimator):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def fit(
        self,
        numerator_samples: np.ndarray,
        denominator_samples: np.ndarray,
    ):
        pass


class KernelDensityRatio(DensityRatio):
    def __init__(
        self,
        regularization: str = "rkhs",
        max_support_points: int = 1000,
    ):
        super().__init__()
        self.regularization = regularization
        self.regularization_weight = 1
        self.max_support_points = max_support_points
        self.dim = None
        self.theta = None
        self.support_points = None
        self.fitted = False

    def __call__(self, w):
        return self.predict(w)

    def predict(self, w):
        assert self.fitted
        if len(w.shape) == 1:
            assert w.shape[0] == self.dim
            w = w.reshape(1, self.dim)
        elif len(w.shape) == 2:
            assert w.shape[1] == self.dim
        else:
            raise ValueError

        return np.maximum((self.kernel(w, self.support_points) @ self.theta).ravel(), 0)
        # return (self.kernel(w, self.support_points) @ self.theta).ravel()


    def kernel(self, w_1: np.ndarray, w_2: np.ndarray):
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
        assert w_1.shape[1] == w_2.shape[1]
        squared_distances = distance_matrix(w_1, w_2)**2
        return np.exp(- self.lengthscale * squared_distances)


    def fit(
        self,
        numerator_samples: np.ndarray,
        denominator_samples: np.ndarray,
        find_regularization_weight = True,
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
            of the estimator.
        max_support_points: int
            Maximum number of numerator samples to use as support points.
            Using too many has severe impacts on computation time during
            training and inference.

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

        numerator_dim = numerator_samples.shape[1]
        denominator_dim = denominator_samples.shape[1]
        condition_dim = (numerator_dim == denominator_dim)
        assert condition_dim, msg_dim

        assert self.regularization in ["l2", "rkhs"], "Unknown regularization"

        if find_regularization_weight:
            best_weight_density_ratio, best_loss_density_ratio = \
                    self.find_best_regularization_weight(
                        numerator_samples,
                        denominator_samples,
                        weights=[10**k for k in range(-3, 3)],
                        max_iter=3,
                    )
            logger.debug(
                f"Best density ratio loss: {best_loss_density_ratio}, " +
                f"with weight {best_weight_density_ratio}"
            )

        median = np.quantile(
            np.ravel(distance_matrix(numerator_samples, numerator_samples)),
            .5
        )
        self.lengthscale = 1 / median**2

        n_samples, self.dim = numerator_samples.shape
        if self.max_support_points <= n_samples:
            self.support_points = numerator_samples[:self.max_support_points]
        else:
            self.support_points = numerator_samples
        n_support_points = self.support_points.shape[0]

        K = self.kernel(self.support_points, numerator_samples)
        h_hat = np.mean(K, axis=1, keepdims=True)
        cross_K = self.kernel(self.support_points, denominator_samples)
        H_hat = cross_K @ cross_K.T / n_samples

        if self.regularization == "l2":
            self.theta = np.linalg.solve(
                H_hat + self.regularization_weight * np.eye(n_support_points), h_hat
            )
        elif self.regularization == "rkhs":
            self.theta = np.linalg.solve(
                H_hat + self.regularization_weight * self.kernel(self.support_points, self.support_points),
                h_hat
            )
        self.fitted = True

    def compute_loss(
        self,
        numerator_samples: np.ndarray,
        denominator_samples: np.ndarray,
    ) -> float:
        assert numerator_samples.shape == denominator_samples.shape
        assert self.fitted
        loss = np.mean(
            np.square(self.predict(denominator_samples))/2
            - self.predict(numerator_samples)
        )
        return loss

    def find_best_regularization_weight(
        self,
        numerator_samples: np.ndarray,
        denominator_samples: np.ndarray,
        n_splits: int = 5,
        weights: list = [10**(-i) for i in range(-2, 3)],
        base_offset = None,
        current_iter = 0,
        max_iter = 2,
    ) -> float:
        """Uses K-Fold cross validation to choose regularization weight.

        Uses a recursion mechanism to better choose the regularization weight.

        """
        assert numerator_samples.shape == denominator_samples.shape
        Kf = KFold(n_splits=n_splits)
        fold_losses_by_weight = {weight: np.empty(n_splits) for weight in weights}
        for weight in weights:
            for fold, (train_idx, test_idx) in enumerate(Kf.split(numerator_samples)):
                numerator_train = numerator_samples[train_idx]
                denominator_train = denominator_samples[train_idx]
                numerator_test = numerator_samples[test_idx]
                denominator_test = denominator_samples[test_idx]
                self.regularization_weight = weight
                self.fit(numerator_train, denominator_train, find_regularization_weight=False)
                loss = self.compute_loss(numerator_test, denominator_test)
                fold_losses_by_weight[weight][fold] = loss
        cv_loss_by_weight = {
            weight: np.mean(losses)
            for weight, losses in fold_losses_by_weight.items()
        }
        best_weight = min(cv_loss_by_weight, key=cv_loss_by_weight.get)
        if current_iter == max_iter:
            best_weight_loss = cv_loss_by_weight[best_weight]
            self.regularization_weight = best_weight
            return best_weight, best_weight_loss
        elif current_iter == 0:
            log_10 = lambda x: np.log(x)/np.log(10)
            base_offset = np.power(10, np.floor(log_10(best_weight)) - 1)
            new_weights = [best_weight + k*base_offset for k in range(-5, 6)]
            return self.find_best_regularization_weight(
                numerator_samples,
                denominator_samples,
                n_splits,
                new_weights,
                base_offset,
                current_iter+1,
                max_iter,
            )
        else:
            new_base_offset = base_offset / 10
            new_weights = [best_weight + k * new_base_offset for k in range(-5, 6)]
            return self.find_best_regularization_weight(
                numerator_samples,
                denominator_samples,
                n_splits,
                new_weights,
                new_base_offset,
                current_iter+1,
                max_iter,
            )

class DeepDensityRatio(DensityRatio):
    def __init__(
        self,
    ):
        super().__init__()


