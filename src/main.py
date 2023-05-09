"""Entry point to run stuff.

Author: @Caioflp

"""
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neighbors import KernelDensity as KDE

from src.data.synthetic import make_low_dimensional_regression
from src.models import FunctionalGD, FunctionalSGD


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(f"Working directory: {os.getcwd()}")


if __name__ == "__main__":
    main()
    # dataset = make_low_dimensional_regression(1000, response="sin")

    # projector_y = KNN(n_neighbors=1, weights="distance")
    # projector_estimate = KNN(n_neighbors=5, weights="distance")
    # regressor_grad = KNN(n_neighbors=5, weights="distance")
    # density_estimator_x = KDE()
    # density_estimator_xz = KDE()
    # density_estimator_z = KDE()

    # model_gd = FunctionalGD(
    #     "inv_sqrt",
    #     projector_y,
    #     projector_estimate,
    #     regressor_grad
    # )
    # model_sgd = FunctionalSGD(
    #     "inv_n_samples",
    #     projector_y,
    #     projector_estimate,
    #     density_estimator_x,
    #     density_estimator_z,
    #     density_estimator_xz,
    # )

    # for model, title in zip(
    #     [model_sgd],#, model_gd],
    #     ["Stochastic Gradient Descent"]#, "Gradient Descent"]
    # ):
    #     model.fit(dataset.X, dataset.Z, dataset.Y)

    #     domain = model.estimate_domain.flatten()
    #     sort_idx = np.argsort(domain)
    #     # plt.plot(domain[sort_idx], model.estimate[sort_idx], c="b", label="model")
    #     fig, axs = plt.subplots(3)
    #     ax_obs = axs[0]
    #     ax_model_on_obs = axs[1]
    #     ax_model_on_grid = axs[2]

    #     ax_obs.scatter(dataset.X, dataset.Y_denoised, c="r", s=4, label="truth")
    #     ax_obs.scatter(dataset.X, dataset.Y, c="y", s=3, label="observed")
    #     ax_obs.legend()
    #     ax_obs.set_title("Data")

    #     ax_model_on_grid.scatter(dataset.X, dataset.Y_denoised, c="r", s=4, label="truth")
    #     ax_model_on_grid.scatter(model.grid_domain, model.estimate_on_grid, c="b", s=3, label="model")
    #     ax_model_on_grid.scatter(model.grid_domain,
    #                              model.sequence_of_estimates.on_grid_points[-1],
    #                              c="k", s=3, label="last estimate")
    #     ax_model_on_grid.legend()
    #     ax_model_on_grid.set_title("Model on grid points")

    #     ax_model_on_obs.scatter(dataset.X, dataset.Y_denoised, c="r", s=4, label="truth")
    #     ax_model_on_obs.scatter(dataset.X, model.estimate_on_obs, c="b", s=3, label="model")
    #     ax_model_on_obs.scatter(dataset.X,
    #                             model.sequence_of_estimates.on_observed_points[-1],
    #                             c="k", s=3, label="last estimate")
    #     ax_model_on_obs.legend()
    #     ax_model_on_obs.set_title("Model on observed points")

    #     fig.suptitle(title)
    #     fig.tight_layout()
    #     # fig.savefig(title.lower().replace(" ", "_") + ".pdf")
    #     plt.show()

