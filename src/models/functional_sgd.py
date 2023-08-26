"""Implements a functional stochastic gradient descent algorithm for IV
problems.

Author: @Caioflp

"""
import logging
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

from src.data.utils import InstrumentalVariableDataset
from src.models import DensityRatio, ConditionalMeanOperator
from src.models.utils import (
    Estimates,
    FinalEstimate,
    Domain,
    create_covering_grid,
    ensure_two_dimensional,
)


class FunctionalSGD(BaseEstimator):
    """Regressor based on a variant of stochastic gradient descent.

    """

    has_discretized_estimate = True

    def __init__(
        self,
        lr: Literal["inv_sqrt", "inv_n_samples"] = "inv_n_samples",
        warm_up_duration: int = 50,
    ):
        self.lr = lr
        self.warm_up_duration = 50

    def fit(self, dataset: InstrumentalVariableDataset) -> None:
        """Fits model to dataset.

        Parameters
        ----------
        dataset: InstrumentalVariableDataset
            Dataset containing X, Z and Y.

        """
        self.fit_dataset_name = dataset.name

        X, Z, Y, Z_loop = dataset.X, dataset.Z, dataset.Y, dataset.Z_loop
        X = ensure_two_dimensional(X)
        Z = ensure_two_dimensional(Z)
        Z_loop = ensure_two_dimensional(Z_loop)

        n_samples = X.shape[0]
        n_iter = n_samples

        lr_dict = {
            "inv_n_samples": lambda i: 1/np.sqrt(n_samples),
            "inv_sqrt": lambda i: 1/np.sqrt(i)
        }
        lr = lr_dict[self.lr]

        # Create domain for estimates. This Domain object contais the observed
        # random points and the unobserved grid points.
        x_domain = Domain(
            observed_points=X,
            grid_points=create_covering_grid(X).reshape(-1, 1)
        )
        n_grid_points = x_domain.grid_points.shape[0]

        # Create object which will store the sequence of estimates evaluated on
        # unobserved grid points and on observed random points.
        estimates = Estimates(
            n_estimates=n_iter+1,
            n_observed_points=n_samples,
            n_grid_points=n_grid_points,
        )
        estimates.on_all_points[0] = np.zeros(n_grid_points + n_samples)

        # Fit DensityRatio Model
        density_ratio = DensityRatio(
            regularization="l2",
            regularization_weight=0.1,
        )
        joint_samples = np.concatenate([X, Z], axis=1)
        independent_samples = np.concatenate(
            [X, np.roll(Z, 1)],
            axis=1,
        )
        density_ratio.fit(join_samples, independent_samples)

        # Fit ConditionalMeanOperator model
        conditional_mean = ConditionalMeanOperator(
            regularization_weight=0.1,
        )
        conditional_mean.loop_fit(Z, Z_loop)

        for i in tqdm(range(n_iter)):
            # Project current estimate on Z, i.e., compute E [Th_{i-1}(X) | Z]
            projected_current_estimate = \
                    conditional_mean.loop_predict(
                        estimates.on_observed_points[i], i
                    )
            # Project Y on current Z
            projected_y = conditional_mean.loop_predict(Y, i)

            pointwise_loss_grad = \
                    projected_current_estimate - projected_y

            # Compute the ratio of densities p(x, z)/(p(x) * p(z)) which
            # appears in the functional gradient expression
            z_i = np.full((n_grid_points + n_samples, Z.shape[1]), Z[i])
            joint_x_and_current_z = np.concatenate(
                (x_domain.all_points, z_i), axis=1
            )
            ratio_of_densities = density_ratio.predict(join_x_and_current_z)

            # Compute the stochastic estimates for the functional loss gradient
            functional_grad = (
                ratio_of_densities * pointwise_loss_grad
            )

            # Take one step in the negative gradient direction
            estimates.on_all_points[i+1] = (
                estimates.on_all_points[i] - lr(i+1) * functional_grad
            )

        # Construct final estimate as average of sequence of estimates
        # Discard the first `self.warm_up_duration` samples if we have enough
        # estimates. If we don't, simply average them all.

        self.sequence_of_estimates = estimates
        if self.warm_up_duration < n_samples:
            self.estimate = FinalEstimate(
                on_observed_points=estimates \
                                   .on_observed_points[self.warm_up_duration:] \
                                   .mean(axis=0),
                on_grid_points=estimates \
                               .on_grid_points[self.warm_up_duration:] \
                               .mean(axis=0),
            )
        else:
            self.estimate = FinalEstimate(
                on_observed_points=estimates.on_observed_points[1:].mean(axis=0),
                on_grid_points=estimates.on_grid_points[1:].mean(axis=0),
            )
        self.domain = x_domain
        self.is_fitted = True
