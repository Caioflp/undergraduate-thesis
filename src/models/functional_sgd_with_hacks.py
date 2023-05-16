"""Implements FunctionalSGD knowing the distribution of the data.

We assume the model will be trained on the poster data.

Author: @Caioflp

"""
import logging
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
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


class FunctionalSGDWithHacks(BaseEstimator):
    """Regressor based on a variant of stochastic gradient descent.

    Knows the data's distribution (hence the Hacks).

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
        use_true_density: bool = True,
        use_true_conditional_expectations: bool = True,
        rho: float = 0.7,
    ):
        self.lr = lr
        self.projector_y = projector_y
        self.projector_estimate = projector_estimate
        self.density_estimator_x = density_estimator_x
        self.density_estimator_z = density_estimator_z
        self.density_estimator_xz = density_estimator_xz
        self.use_true_density = use_true_density
        self.use_true_conditional_expectations = \
                use_true_conditional_expectations
        self.rho = rho

    def fit(self, dataset: InstrumentalVariableDataset) -> None:
        """Fits model to dataset.

        Parameters
        ----------
        dataset: InstrumentalVariableDataset
            Dataset containing X, Z and T.

        """
        self.fit_dataset_name = dataset.name
        assert dataset.name == "poster dataset"

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

        if not self.use_true_density:
            # Fit the density estimators on X, Z and (X, Z)
            densities_x = self.density_estimator_x.fit(X) \
                    .score_samples(x_domain.all_points)
            densities_z = self.density_estimator_z.fit(Z).score_samples(Z)
            self.density_estimator_xz.fit(np.concatenate((X, Z), axis=1))

        for i in tqdm(range(n_iter)):
            # Project current estimate on Z, i.e., compute E [h_{i-1}(X) | Z]
            projected_current_estimate = self.projector_estimate \
                                         .fit(Z, estimates.on_observed_points[i]) \
                                         .predict([Z[i]])[0]

            pointwise_loss_grad = \
                    projected_current_estimate - projected_y[i]

            # Compute the ratio of densities p(x, z)/(p(x) * p(z)) which
            # appears in the functional gradient expression
            z_i = np.full((n_grid_points + n_samples, Z.shape[1]), Z[i])
            if self.use_true_density:
                # The marginals are one, so the ratio is just the joint density
                # (which happens to also be written as a ratio).
                # We make computations in the log scale and exponentiate the
                # result.
                transformed_x = scipy.stats.norm.ppf(x_domain.all_points)
                transformed_z = scipy.stats.norm.ppf(z_i)
                cov = np.array([[1, self.rho], [self.rho, 1]])
                mean = np.zeros(2, dtype=np.float64)
                log_numerator = scipy.stats.multivariate_normal.logpdf(
                    np.concatenate((transformed_x, transformed_z), axis=1),
                    mean=mean, cov=cov
                )
                log_denominator = (
                    scipy.stats.norm.logpdf(transformed_x)
                    + scipy.stats.norm.logpdf(transformed_z)
                ).flatten()
                ratio_of_densities = np.exp(log_numerator - log_denominator)
            else:
                # We apply the exponential because `score_samples` returns
                # log densities.
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
        self.sequence_of_estimates = estimates
        self.estimate = FinalEstimate(
            on_observed_points=estimates.on_observed_points[1:].mean(axis=0),
            on_grid_points=estimates.on_grid_points[1:].mean(axis=0),
        )
        self.domain = x_domain
        self.is_fitted = True
