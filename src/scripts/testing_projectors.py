"""Script to test different projectors

Author: Caio

"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KernelDensity as KDE
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.linear_model import LinearRegression

from src.data import InstrumentalVariableDataset
from src.data.synthetic import (
    make_poster_dataset,
    make_deep_gmm_dataset,
)
from src.models import FunctionalSGD
from src.scripts.utils import experiment


def make_model(
    use_linear_regression: bool = False,
    n_neighbors: int = 5,
    weights: str = "distance",
    bandwidth_x: float = 0.1,
    bandwidth_z: float = 0.1,
    bandwidth_xz: float = 0.1,
) -> FunctionalSGD:
    if use_linear_regression:
        projector_y = LinearRegression()
        projector_estimate = LinearRegression()
    else:
        projector_y = KNN(n_neighbors=n_neighbors)
        projector_estimate = KNN(n_neighbors=n_neighbors)
    density_estimator_x = KDE(bandwidth=bandwidth_x)
    density_estimator_z = KDE(bandwidth=bandwidth_z)
    density_estimator_xz = KDE(bandwidth=bandwidth_xz)
    model = FunctionalSGD(
        projector_y=projector_y,
        projector_estimate=projector_estimate,
        density_estimator_x=density_estimator_x,
        density_estimator_z=density_estimator_z,
        density_estimator_xz=density_estimator_xz,
    )
    return model


def plot_estimate(
    model: FunctionalSGD,
    dataset: InstrumentalVariableDataset,
    figsize: Tuple[int] = (7, 5),
    with_data = True,
    title: str = "Estimate",
) -> None:
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)
    # Sorting is necessary to make line plots
    sorted_idx_observed = np.argsort(model.domain.observed_points.flatten())
    ax.plot(
        dataset.X.flatten()[sorted_idx_observed],
        dataset.Y_denoised[sorted_idx_observed],
        c="r",
        label="Denoised response",
        alpha=.8,
    )
    sort_idx = np.argsort(model.domain.all_points.flatten())
    sorted_x = model.domain.all_points.flatten()[sort_idx]
    sorted_estimate = model.estimate.on_all_points[sort_idx]
    sorted_last_estimate = \
            model.sequence_of_estimates.on_all_points[-1][sort_idx]
    ax.plot(
        sorted_x,
        sorted_estimate,
        c="b",
        label="Average estimate",
    )
    ax.plot(
        sorted_x,
        sorted_last_estimate,
        c="k",
        label="Last estimate",
    )
    if with_data:
        ax.scatter(
            dataset.X.flatten(),
            dataset.Y,
            c="k",
            s=2,
            alpha=0.6,
            label="Observed response",
        )

    ax.set_title(title)
    ax.legend()
    fig.savefig(title.lower().replace(" ", "_") + ".pdf")


def test_knn_vs_lin_reg():
    for response in [f"case_{i}" for i in range(1, 4)]:
        dataset = make_poster_dataset(response=response)
        model_lin_reg = make_model(use_linear_regression=True)
        model_knn = make_model()
        print(f"Fitting model with linear regression for {response}")
        model_lin_reg.fit(dataset)
        print(f"Fitting model with knn for {response}")
        model_knn.fit(dataset)
        plot_estimate(model_knn, dataset, title=f"{response} knn")
        plot_estimate(model_lin_reg, dataset, title=f"{response} linear regression")


def test_different_k_for_knn():
    for response in [f"case_{i}" for i in range(1, 4)]:
        for k in [5*i for i in range(1, 5)]:
            dataset = make_poster_dataset(response=response)
            model = make_model(n_neighbors=k)
            print(f"Fitting model with {k} neighbors for {response}")
            model.fit(dataset)
            plot_estimate(model, dataset, title=f"{k} neighbors {response}")


@experiment("testing_projectors/knn_with_different_k")
def main():
    test_different_k_for_knn()


if __name__ == "__main__":
    main()
