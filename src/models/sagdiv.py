"""Implements the algorithm 'Stochastic Approximate Gradient Descent IV'.

Author: @Caioflp

"""
import logging
from time import time
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

from src.data.utils import InstrumentalVariableDataset
from src.models import DensityRatio, ConditionalMeanOperator
from src.models.utils import (
    Loss,
    QuadraticLoss,
    ensure_two_dimensional,
    truncate,
)


logger = logging.getLogger("src.model")


class SAGDIV(BaseEstimator):
    """Regressor based on a variant of stochastic gradient descent.

    """
    def __init__(
        self,
        lr: Literal["inv_sqrt", "inv_n_samples"] = "inv_n_samples",
        loss: Loss = QuadraticLoss(),
        warm_up_duration: int = 50,
        bound: float = 10,
        update_scheme: Literal["sgd", "nesterov"] = "sgd",
    ):
        self.lr = lr
        self.loss = loss
        self.warm_up_duration = 50
        self.bound = bound
        self.update_scheme = update_scheme
        self.is_fitted = False
        self.fit_dataset_name = None
        self.lr_func = None
        self.density_ratio_model = None
        self.conditional_mean_model_xz = None
        self.conditional_mean_model_yz = None
        self.loss_derivative_array = None
        self.sequence_of_estimates = None
        self.estimate = None
        self.domain = None
        self.Z_loop = None

    def fit_density_ratio_model(
        self,
        X: np.ndarray,
        Z: np.ndarray,
    ) -> None:
        """ Fits density ratio model.

        """
        start = time()
        self.density_ratio_model = DensityRatio(regularization="rkhs")
        joint_samples = np.concatenate([X, Z], axis=1)
        independent_samples = np.concatenate([X, np.roll(Z, 2, axis=0)], axis=1)
        best_weight_density_ratio, best_loss_density_ratio = \
                self.density_ratio_model.find_best_regularization_weight(
                    joint_samples,
                    independent_samples,
                    weights=[10**k for k in range(-3, 3)],
                    max_iter=3,
                )
        logger.debug(
            f"Best density ratio loss: {best_loss_density_ratio}, " +
            f"with weight {best_weight_density_ratio}"
        )
        self.density_ratio_model.fit(joint_samples, independent_samples)
        end = time()
        logger.info("Density ratio model fitted.")
        logger.debug(f"Time to fit density ratio model: {end-start:1.2e}s")

    def compute_density_ratios(
        self,
        X: np.ndarray,
        Z_loop: np.ndarray,
    ) -> np.ndarray:
        """ Computes all necessary density ratio evaluations for
        evaluating/fitting the SAGD-IV estimator on some `X` and `Z_loop` samples.

        """
        start = time()
        n_samples = X.shape[0]
        n_iter = Z_loop.shape[0]
        dim_z = Z_loop.shape[1]
        dim_x = X.shape[1]
        repeated_z_samples = np.full(
            (n_samples, *Z_loop.shape),
            Z_loop,
        )
        repeated_z_samples = repeated_z_samples \
                             .transpose((1, 0, 2)) \
                             .reshape(
                                 (n_iter*n_samples, dim_z)
                             )
        repeated_x_points = np.full(
            (n_iter, *X.shape),
            X,
        )
        repeated_x_points = repeated_x_points.reshape(
            (n_iter*n_samples, dim_x)
        )
        joint_x_and_all_z = np.concatenate(
            (repeated_x_points, repeated_z_samples),
            axis=1
        )
        density_ratios = self.density_ratio_model.predict(joint_x_and_all_z)
        density_ratios = density_ratios.reshape((n_iter, n_samples))
        end = time()
        logger.debug(f"Time to pre-compute density ratios: {end-start:1.2e}s")
        logger.info("Density ratios pre-computed.")
        return density_ratios

    def fit_conditional_mean_models(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        Y: np.ndarray,
        Z_loop: np.ndarray,
    ) -> None:
        """ Fits models for estimating the conditional expectation operators of
        X conditioned on Z and Y conditioned on Z.

        """
        self.conditional_mean_model_xz = ConditionalMeanOperator()
        best_weight_xz, best_loss_xz = \
                self.conditional_mean_model_xz.find_best_regularization_weight(Z, X)
        logger.debug(
            f"Best conditional mean XZ loss: {best_loss_xz}, with weight "
            + f"{best_weight_xz}"
        )
        self.conditional_mean_model_xz.loop_fit(Z, Z_loop)
        logger.info("Conditional Mean Operator of X|Z fitted.")

        Y_array = Y.reshape(-1, 1)
        self.conditional_mean_model_yz = ConditionalMeanOperator()
        best_weight_yz, best_loss_yz = \
                self.conditional_mean_model_yz.find_best_regularization_weight(Z, Y_array)
        logger.debug(
            f"Best conditional mean YZ loss: {best_loss_yz}, with weight " +
            f"{best_weight_yz}"
        )
        self.conditional_mean_model_yz.loop_fit(Z, Z_loop)
        logger.info("Conditional Mean Operator of Y|Z fitted.")


    def fit(self, dataset: InstrumentalVariableDataset) -> None:
        """Fits model to dataset.

        Parameters
        ----------
        dataset: InstrumentalVariableDataset
            Dataset containing X, Z and Y.

        """
        fit_start = time()

        self.fit_dataset_name = dataset.name

        X, Z, Y, Z_loop = dataset.X, dataset.Z, dataset.Y, dataset.Z_loop
        X = ensure_two_dimensional(X)
        Z = ensure_two_dimensional(Z)
        Z_loop = ensure_two_dimensional(Z_loop)

        n_samples = X.shape[0]
        n_iter = Z_loop.shape[0]
        dim_x = X.shape[1]
        dim_z = Z.shape[1]

        lr_dict = {
            "inv_n_samples": lambda i: 1/np.sqrt(n_samples),
            "inv_sqrt": lambda i: 1/np.sqrt(i)
        }
        self.lr_func = lr_dict[self.lr]

        # Create array which will store the gradient descent path of estimates
        # `n_iter+1` is due to the first estimate, which is the null function
        estimates = np.zeros((n_iter+1, n_samples), dtype=float)

        self.fit_density_ratio_model(X, Z)
        # density_ratios = self.compute_density_ratios(X, Z_loop)

        self.fit_conditional_mean_models(X, Z, Y, Z_loop)

        # Create array for storing \partial_{2} \ell values for later
        # calls to `predict`
        self.loss_derivative_array = np.empty(n_iter, dtype=np.float64)

        if self.update_scheme == "nesterov":
            momentum = 1/n_samples
            phi_current = estimates.on_all_points[0]

        execution_times = {
            "computing conditional expectations": [],
            "computing pointwise loss gradient": [],
            "computing ratio of densities": [],
            "computing functional gradient": [],
            "computing projected gradient descent step": [],
        }

        for i in tqdm(range(n_iter)):
            start = time()
            # Project current estimate on Z, i.e., compute E [h_{i-1}(X) | Z]
            projected_current_estimate = \
                    self.conditional_mean_model_xz.loop_predict(estimates[i], i)
            # Project Y on current Z
            projected_y = self.conditional_mean_model_yz.loop_predict(Y, i)
            end = time()
            execution_times["computing conditional expectations"].append(end-start)

            start = time()
            loss_derivative = self.loss.derivative_second_argument(
                projected_y,
                projected_current_estimate,
            )
            end = time()
            self.loss_derivative_array[i] = loss_derivative
            execution_times["computing pointwise loss gradient"].append(end-start)

            start = time()
            # ratio_of_densities = density_ratios[i]
            joint_x_and_z_i = np.concatenate(
                [X, np.full((n_samples, dim_z), Z_loop[i])],
                axis=1
            )
            ratio_of_densities = self.density_ratio_model.predict(joint_x_and_z_i)
            end = time()
            execution_times["computing ratio of densities"].append(end-start)

            # Compute the stochastic estimates for the functional loss gradient
            start = time()
            stochastic_approximate_gradient = (
                ratio_of_densities * loss_derivative
            )
            end = time()
            execution_times["computing functional gradient"].append(end-start)

            # Take one step in the negative gradient direction
            start = time()
            sagd_update = (
                estimates[i] - self.lr_func(i+1)*stochastic_approximate_gradient
            )
            if self.update_scheme == "nesterov":
                phi_next = sagd_update
                estimates[i+1] = truncate(
                    phi_next + momentum*(phi_next - phi_current),
                    self.bound
                )
                phi_current = phi_next
            else:
                estimates[i+1] = truncate(sagd_update, self.bound)
            end = time()
            execution_times["computing projected gradient descent step"].append(end-start)

        for action, times in execution_times.items():
            mean = np.mean(times)
            std = np.std(times)
            logger.debug(f"Average time spent {action}: {mean:1.2e}s±{std:1.2e}s")

        # Save Z_loop values for predict method
        if self.warm_up_duration < n_samples:
            self.Z_loop = Z_loop[self.warm_up_duration:]
            self.loss_derivative_array = self.loss_derivative_array[self.warm_up_duration:]
        else:
            self.Z_loop = Z_loop
        self.is_fitted = True

        fit_end = time()
        logger.debug(f"Time spent fitting SAGD-IV model: {fit_end-fit_start:1.2e}s")

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted
        # Compute all necessary density ratios
        X = ensure_two_dimensional(X)
        n_iter = self.Z_loop.shape[0]
        density_ratios = self.compute_density_ratios(X, self.Z_loop)
        stochastic_approximate_gradients = \
                density_ratios * self.loss_derivative_array.reshape(-1, 1)
        estimates = np.zeros((n_iter+1, X.shape[0]), dtype=np.float64)
        if self.update_scheme == "nesterov":
            momentum = 1/n_samples
            phi_current = result
        for i, grad in enumerate(stochastic_approximate_gradients):
            sagd_update = \
                    estimates[i] - self.lr_func(i+self.warm_up_duration+1)*grad
            # sagd_update = \
            #         estimates[i] - self.lr_func(i+1)*grad
            if self.update_scheme == "nesterov":
                phi_next = gd_update
                estimates[i+1] = truncate(
                    phi_next + momentum*(phi_next - phi_current),
                    self.bound
                )
                phi_current = phi_next
            else:
                estimates[i+1] = truncate(sagd_update, self.bound)
        return np.mean(estimates, axis=0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)
