"""Script to plot density estimates with hacked model

Author: Caio

"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KernelDensity as KDE
from sklearn.neighbors import KNeighborsRegressor as KNN

from src.data import InstrumentalVariableDataset
from src.data.synthetic import (
    make_poster_dataset,
    make_deep_gmm_dataset,
)
from src.models import FunctionalSGDWithHacks
from src.scripts.utils import experiment


def make_model(
    n_neighbors: int = 5,
    weights: str = "distance",
    bandwidth_x: float = 1.0,
    bandwidth_z: float = 1.0,
    bandwidth_xz: float = 1.0,
) -> FunctionalSGDWithHacks:
    projector_y = KNN(n_neighbors=n_neighbors)
    projector_estimate = KNN(n_neighbors=n_neighbors)
    density_estimator_x = KDE(bandwidth=bandwidth_x)
    density_estimator_z = KDE(bandwidth=bandwidth_z)
    density_estimator_xz = KDE(bandwidth=bandwidth_xz)
    model = FunctionalSGDWithHacks(
        projector_y=projector_y,
        projector_estimate=projector_estimate,
        density_estimator_x=density_estimator_x,
        density_estimator_z=density_estimator_z,
        density_estimator_xz=density_estimator_xz,
        use_true_density=True,
        use_true_conditional_expectations=False,
    )
    return model


def plot_estimate(
    model: FunctionalSGDWithHacks,
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
    ax.set_xlim([0.05, 0.95])
    ax.legend()
    fig.savefig(title.lower().replace(" ", "_") + ".pdf")



@experiment("model_with_hacks/")
def main():
    for response in [f"case_{i}" for i in range(1, 3+1)]:
        n_samples = 700
        dataset = make_poster_dataset(n_samples=n_samples, response=response)
        h = 0.1
        model = make_model(
            bandwidth_x=h,
            bandwidth_z=h,
            bandwidth_xz=h,
        )
        model.fit(dataset)
        plot_estimate(
            model, dataset,
            title=f"Estimate {response} true density n_samples {n_samples}"
        )


if __name__ == "__main__":
    main()
