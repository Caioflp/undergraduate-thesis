"""Entry point to run stuff.

Author: @Caioflp

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor as KNN

from src.data.synthetic import make_low_dimensional_regression
from src.models import FunctionalGradientDescentIV


if __name__ == "__main__":
    dataset = make_low_dimensional_regression(200, response="sin")

    projector_y = KNN(n_neighbors=1, weights="distance")
    projector_estimate = KNN(n_neighbors=5, weights="distance")
    regressor_grad = KNN(n_neighbors=5, weights="distance")

    model = FunctionalGradientDescentIV(
        "inv_sqrt",
        projector_y,
        projector_estimate,
        regressor_grad
    )
    model.fit(dataset.X, dataset.Z, dataset.Y)

    domain = model.estimate_domain.flatten()
    sort_idx = np.argsort(domain)
    # plt.plot(domain[sort_idx], model.estimate[sort_idx], c="b", label="model")
    fig, axs = plt.subplots(3)
    ax_obs = axs[0]
    ax_model_on_obs = axs[1]
    ax_model_on_grid = axs[2]

    ax_obs.scatter(dataset.X, dataset.Y_denoised, c="r", s=4, label="truth")
    ax_obs.scatter(dataset.X, dataset.Y, c="y", s=3, label="observed")
    ax_obs.legend()
    ax_model_on_grid.set_title("Data")

    ax_model_on_grid.scatter(dataset.X, dataset.Y_denoised, c="r", s=4, label="truth")
    ax_model_on_grid.scatter(model.grid_domain, model.estimate_on_grid, c="b", s=3, label="model")
    ax_model_on_grid.legend()
    ax_model_on_grid.set_title("Model on grid points")

    ax_model_on_obs.scatter(dataset.X, dataset.Y_denoised, c="r", s=4, label="truth")
    ax_model_on_obs.scatter(dataset.X, model.estimate_on_obs, c="b", s=3, label="model")
    ax_model_on_obs.legend()
    ax_model_on_grid.set_title("Model on observed points")

    fig.tight_layout()
    plt.show()

