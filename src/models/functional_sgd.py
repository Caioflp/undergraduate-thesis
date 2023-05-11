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
    DEFAULT_REGRESSOR,
    DEFAULT_DENSITY_ESTIMATOR,
    Estimates,
    FinalEstimate,
    Domain,
    create_covering_grid,
    ensure_two_dimensional,
)


LOGGER = logging.getLogger(__name__)


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
    def __init__(
        self,
        lr: Literal["inv_sqrt", "inv_n_samples"] = "inv_n_samples",
        projector_y: BaseEstimator = DEFAULT_REGRESSOR,
        projector_estimate: BaseEstimator = DEFAULT_REGRESSOR,
        density_estimator_x: BaseEstimator = DEFAULT_DENSITY_ESTIMATOR,
        density_estimator_z: BaseEstimator = DEFAULT_DENSITY_ESTIMATOR,
        density_estimator_xz: BaseEstimator = DEFAULT_DENSITY_ESTIMATOR,
    ):
        self.lr = lr
        self.projector_y = projector_y
        self.projector_estimate = projector_estimate
        self.density_estimator_x = density_estimator_x
        self.density_estimator_z = density_estimator_z
        self.density_estimator_xz = density_estimator_xz

    def fit(self, dataset: InstrumentalVariableDataset) -> None:
        """Fits model to dataset.

        Parameters
        ----------
        dataset: InstrumentalVariableDataset
            Dataset containing X, Z and T.

        """
        LOGGER.info("Fitting model.")
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
        # densities_x_observed = self.density_estimator_x.fit(X) \
        #                                                .score_samples(X)
        # densities_x_grid = self.density_estimator_x.score_samples(x_domain)
        densities_z = self.density_estimator_z.fit(Z).score_samples(Z)
        self.density_estimator_xz.fit(np.concatenate((X, Z), axis=1))

        for i in tqdm(range(n_iter)):
            # Project current estimate on Z, i.e., compute E [Th(X) | Z]
            projected_current_estimate = self.projector_estimate \
                                         .fit(Z, estimates.on_observed_points[i]) \
                                         .predict([Z[i]])[0]

            # Compute the the gradient of the pointwise loss function.
            # We are assuming that the loss function is quadratic,
            # which makes its gradient linear.
            # TODO: Implement other types of loss functions.
            pointwise_loss_grad = \
                    projected_current_estimate - projected_y[i]

            # Compute the ratio of densities p(x, z)/(p(x) * p(z)) which
            # appears in the functional gradient expression
            # We apply the exponential because `score_samples` returns
            # log densities.
            joint_x_and_current_z = np.concatenate(
                (
                    x_domain.all_points,
                    np.full((n_grid_points + n_samples, Z.shape[1]), Z[i])
                ),
                axis=1
            )
            ratio_of_densities = np.exp(
                self.density_estimator_xz.score_samples(joint_x_and_current_z)
                - densities_x
                - densities_z[i]
            )

            # ratio_of_densities_grid = np.exp(
            #     self.density_estimator_xz.score_samples(
            #         np.hstack((
            #             x_domain,
            #             np.full((n_grid_points, Z.shape[1]), Z[i])
            #         ))
            #     )
            #     - densities_x_grid
            #     - densities_z[i]
            # )
            # ratio_of_densities_observed = np.exp(
            #     self.density_estimator_xz.score_samples(
            #         np.hstack((
            #             X, np.full((n_samples, Z.shape[1]), Z[i])
            #         ))
            #     )
            #     - densities_x_observed
            #     - densities_z[i]
            # )

            # Compute the stochastic estimates for the functional loss gradient
            functional_grad = (
                ratio_of_densities * pointwise_loss_grad
            )

            # functional_grad_grid = (
            #     ratio_of_densities_grid * pointwise_loss_grad
            # )
            # functional_grad_observed = (
            #     ratio_of_densities_observed * pointwise_loss_grad
            # )

            # Take one step in the negative gradient direction
            estimates.on_all_points[i+1] = (
                estimates.on_all_points[i]
                - lr(i+1) * functional_grad
            )
            # estimates.on_grid_points[i+1, :] = (
            #     estimates.on_grid_points[i, :]
            #     - lr(i+1) * functional_grad_grid
            # )
            # estimates.on_observed_points[i+1, :] =  (
            #     estimates.on_observed_points[i, :]
            #     - lr(i+1) * functional_grad_observed
            # )

        # Construct final estimate as average of sequence of estimates
        self.sequence_of_estimates = estimates
        self.estimate = FinalEstimate(
            on_observed_points=estimates.on_observed_points.mean(axis=0),
            on_grid_points=estimates.on_grid_points.mean(axis=0),
        )
        self.domain = x_domain

        # mean_estimate_on_grid = estimates.on_grid_points.mean(axis=0)
        # mean_estimate_on_observed = estimates.on_observed_points.mean(axis=0)
        # self.estimate_domain = np.concatenate((x_domain, X), axis=0)
        # self.estimate = np.concatenate(
        #     (mean_estimate_on_grid, mean_estimate_on_observed),
        #     axis=None
        # )
        # self.grid_domain = x_domain
        # self.estimate_on_grid = mean_estimate_on_grid
        # self.estimate_on_obs = mean_estimate_on_observed
        # self.sequence_of_estimates = estimates
        self.is_fitted = True

    def plot_data(
        self,
        dataset: InstrumentalVariableDataset,
        figsize: Tuple[int] = (7, 7),
    ) -> None:
        fig, ax = plt.subplots(layout="constrained", figsize=figsize)
        title = "Data"
        ax.scatter(
            self.domain.observed_points.flatten(),
            dataset.Y,
            c="y",
            s=1.5,
            alpha=0.8,
            label="Observed response",
        )
        sort_idx = np.argsort(self.domain.observed_points)
        sorted_x = self.domain.observed_points[sort_idx]
        sorted_y_denoised = dataset.Y_denoised[sort_idx]
        print(sorted_y_denoised)
        ax.plot(
            sorted_x.flatten(),
            sorted_y_denoised,
            c="r",
            label="Denoised response",
        )
        ax.set_title(title)
        ax.legend()
        fig.savefig(title.lower() + ".pdf")

    def plot_estimate_on_observed_points(
        self,
        dataset: InstrumentalVariableDataset,
        figsize: Tuple[int] = (7, 7),
    ) -> None:
        fig, ax = plt.subplots(layout="constrained", figsize=figsize)
        title = "Estimate on observed points"
        sort_idx = np.argsort(self.domain.observed_points)
        sorted_x = self.domain.observed_points[sort_idx]
        sorted_y_denoised = dataset.Y_denoised[sort_idx]
        sorted_estimate = self.estimate.on_observed_points[sort_idx]
        sorted_last_estimate = \
                self.sequence_of_estimates.on_observed_points[-1][sort_idx]
        ax.plot(
            sorted_x.flatten(),
            sorted_y_denoised,
            c="r",
            label="Denoised response",
        )
        ax.plot(
            sorted_x.flatten(),
            sorted_estimate,
            c="b",
            label="Average estimate",
        )
        ax.plot(
            sorted_x.flatten(),
            sorted_last_estimate,
            c="k",
            label="Last estimate",
        )
        ax.set_title(title)
        ax.legend()
        fig.savefig(title.lower().replace(" ", "_") + ".pdf")

    def plot_estimate_on_grid_points(
        self,
        dataset: InstrumentalVariableDataset,
        figsize: Tuple[int] = (7, 7),
    ) -> None:
        fig, ax = plt.subplots(layout="constrained", figsize=figsize)
        title = "Estimate on grid points"
        sort_idx = np.argsort(self.domain.grid_points)
        sorted_x = self.domain.grid_points[sort_idx]
        sorted_y_denoised = dataset.Y_denoised[sort_idx]
        sorted_estimate = self.estimate.on_grid_points[sort_idx]
        sorted_last_estimate = \
                self.sequence_of_estimates.on_grid_points[-1][sort_idx]
        ax.plot(
            sorted_x.flatten(),
            sorted_y_denoised,
            c="r",
            label="Denoised response",
        )
        ax.plot(
            sorted_x.flatten(),
            sorted_estimate,
            c="b",
            label="Average estimate",
        )
        ax.plot(
            sorted_x.flatten(),
            sorted_last_estimate,
            c="k",
            label="Last estimate",
        )
        ax.set_title(title)
        ax.legend()
        fig.savefig(title.lower().replace(" ", "_") + ".pdf")

    def make_plots(self, dataset: InstrumentalVariableDataset) -> None:
        """Makes plots useful for analysing results

        Parameters
        ----------
        dataset: InstrumentalVariableDataset
            Dataset used to fit the model.

        Raises
        ------
        AssertionError if the model was not fitted before calling this method,
        or if the supplied dataset does not match the dataset used during fit.

        """
        assert self.is_fitted
        assert self.fit_dataset_name == dataset.name
        LOGGER.info("Making plots.")

        self.plot_data(dataset)
        self.plot_estimate_on_observed_points(dataset)
        self.plot_estimate_on_grid_points(dataset)

        # title = f"Stochastic Gradient Descent on {dataset.name}"

        # domain = self.estimate_domain.flatten()
        # sort_idx = np.argsort(domain)

        # fig, axs = plt.subplots(3)
        # ax_obs = axs[0]
        # ax_model_on_obs = axs[1]
        # ax_model_on_grid = axs[2]

        # ax_obs.scatter(dataset.X, dataset.Y_denoised, c="r", s=2, label="truth")
        # ax_obs.scatter(dataset.X, dataset.Y, c="y", s=1.5, label="observed")
        # ax_obs.legend()
        # ax_obs.set_title("Data")

        # ax_model_on_grid.scatter(dataset.X, dataset.Y_denoised, c="r", s=2, label="truth")
        # ax_model_on_grid.scatter(self.grid_domain, self.estimate_on_grid, c="b", s=1.5, label="model")
        # ax_model_on_grid.scatter(self.grid_domain,
        #                          self.sequence_of_estimates.on_grid_points[-1],
        #                          c="k", s=1.5, label="last estimate")
        # ax_model_on_grid.legend()
        # ax_model_on_grid.set_title("Model on grid points")

        # ax_model_on_obs.scatter(dataset.X, dataset.Y_denoised, c="r", s=2, label="truth")
        # ax_model_on_obs.scatter(dataset.X, self.estimate_on_obs, c="b", s=1.5, label="model")
        # ax_model_on_obs.scatter(dataset.X,
        #                         self.sequence_of_estimates.on_observed_points[-1],
        #                         c="k", s=1.5, label="last estimate")
        # ax_model_on_obs.legend()
        # ax_model_on_obs.set_title("Model on observed points")

        # fig.suptitle(title)
        # fig.tight_layout()
        # fig.savefig(title.lower().replace(" ", "_") + ".pdf")
