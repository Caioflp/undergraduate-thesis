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
from src.models.utils import (
    Estimates,
    FinalEstimate,
    Domain,
    create_covering_grid,
    ensure_two_dimensional,
    default_regressor,
    default_density_estimator,
)


class FunctionalSGD(BaseEstimator):
    """Regressor based on a variant of stochastic gradient descent.

    Parameters
    ----------
    projector_y: BaseEstimator, default KNN.
        Computes an estimate to E[Y|Z].
    projector_estimate: BaseEstimator, default KNN.
        Computes an estimate to E[h_i-1|Z].
    density_estimator_x: BaseEstimator, default KDE.
        Computes an estimate to p(x).
    density_estimator_z: BaseEstimator, default KDE.
        Computes an estimate to  p(z)
    density_estimator_xz: BaseEstimator, default KDE.
        Computes an estimate to p(x, z)

    """

    has_discretized_estimate = True

    def __init__(
        self,
        lr: Literal["inv_sqrt", "inv_n_samples"] = "inv_n_samples",
        projector_y: BaseEstimator = default_regressor(),
        projector_estimate: BaseEstimator = default_regressor(),
        density_estimator_x: BaseEstimator = default_density_estimator(),
        density_estimator_z: BaseEstimator = default_density_estimator(),
        density_estimator_xz: BaseEstimator = default_density_estimator(),
        warm_up_duration: int = 50,
    ):
        self.lr = lr
        self.projector_y = projector_y
        self.projector_estimate = projector_estimate
        self.density_estimator_x = density_estimator_x
        self.density_estimator_z = density_estimator_z
        self.density_estimator_xz = density_estimator_xz
        self.warm_up_duration = 50

    def fit(self, dataset: InstrumentalVariableDataset) -> None:
        """Fits model to dataset.

        Parameters
        ----------
        dataset: InstrumentalVariableDataset
            Dataset containing X, Z and T.

        """
        self.fit_dataset_name = dataset.name

        X, Z, Y = dataset.X, dataset.Z, dataset.Y
        X = ensure_two_dimensional(X)
        Z = ensure_two_dimensional(Z)

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

        # Compute the projected values of Y onto Z.
        projected_y = self.projector_y.fit(Z, Y).predict(Z)

        # Fit the density estimators on X, Z and (X, Z)
        densities_x = self.density_estimator_x.fit(X) \
                .score_samples(x_domain.all_points)
        densities_z = self.density_estimator_z.fit(Z).score_samples(Z)
        self.density_estimator_xz.fit(np.concatenate((X, Z), axis=1))

        for i in tqdm(range(n_iter)):
            # Project current estimate on Z, i.e., compute E [Th_{i-1}(X) | Z]
            projected_current_estimate = self.projector_estimate \
                                         .fit(Z, estimates.on_observed_points[i]) \
                                         .predict([Z[i]])[0]

            pointwise_loss_grad = \
                    projected_current_estimate - projected_y[i]

            # Compute the ratio of densities p(x, z)/(p(x) * p(z)) which
            # appears in the functional gradient expression
            # We apply the exponential because `score_samples` returns
            # log densities.
            z_i = np.full((n_grid_points + n_samples, Z.shape[1]), Z[i])
            joint_x_and_current_z = np.concatenate(
                (x_domain.all_points, z_i), axis=1
            )
            ratio_of_densities = np.exp(
                self.density_estimator_xz.score_samples(joint_x_and_current_z)
                - densities_x
                - densities_z[i]
            )

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
                on_observed_points=estimates.on_observed_points[1:] .mean(axis=0),
                on_grid_points=estimates.on_grid_points[1:].mean(axis=0),
            )
        self.domain = x_domain
        self.is_fitted = True
