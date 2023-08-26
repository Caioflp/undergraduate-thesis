"""Script to compile everything that was done so far.

Date: 31/05/2023

Author: @Caioflp

"""
from pathlib import Path
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
from src.models import FunctionalSGD
from src.scripts.utils import experiment


def make_model(
    n_neighbors: int = 5,
    weights: str = "distance",
    bandwidth_x: float = 0.1,
    bandwidth_z: float = 0.1,
    bandwidth_xz: float = 0.1,
    warm_up_duration: int = 50,
) -> FunctionalSGD:
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
        warm_up_duration=warm_up_duration,
    )
    return model


def plot_estimate(
    model: FunctionalSGD,
    dataset: InstrumentalVariableDataset,
    figsize: Tuple[int] = (7, 5),
    with_data = True,
    title: str = "Estimate",
    path: Path = None,
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
    if path is None:
        fig.savefig(title.lower().replace(" ", "_") + ".pdf")
    else:
        fig.savefig(path)


def plot_density_estimates(
    model: FunctionalSGD,
    dataset: InstrumentalVariableDataset,
    figsize: Tuple[int] = (7, 5),
    with_data = True,
    plot_x: bool = True,
    plot_z: bool = True,
    title: str = "Densities",
    path: Path = None,
) -> None:
    assert dataset.name == "poster dataset"
    fig, ax = plt.subplots(figsize=figsize)
    if plot_x:
        grid_x = np.linspace(-0.5, 1.5, num=200).reshape(-1, 1)
        model_densities_x = \
                np.exp(model.density_estimator_x.score_samples(grid_x))
        ax.plot(
            grid_x.flatten(),
            model_densities_x,
            c="b",
            label="Model estimate for p(x)"
        )
    if plot_z:
        grid_z = np.linspace(-0.5, 1.5, num=200).reshape(-1, 1)
        model_densities_z = \
                np.exp(model.density_estimator_z.score_samples(grid_z))
        ax.plot(
            grid_z.flatten(),
            model_densities_z,
            c="r",
            label="Model estimate for p(z)"
        )
    ax.set_ylim([0, 1.2])
    ax.set_title(title)
    ax.legend()
    if path is None:
        fig.savefig(title.lower().replace(" ", "_") + ".pdf")
    else:
        fig.savefig(path)


def run_model_on_all_datasets() -> None:
    dataset_makers = [make_deep_gmm_dataset, make_poster_dataset]
    maker_to_responses_dict = {
        make_deep_gmm_dataset: ["sin", "step", "abs", "linear"],
        make_poster_dataset: ["case_1", "case_2", "case_3"],
    }
    for maker in dataset_makers:
        for count, response in enumerate(maker_to_responses_dict[maker]):
            dataset = maker(n_samples=500, response=response)
            h = 0.1 if dataset.name == "poster dataset" else 0.4
            model = make_model(
                bandwidth_x=h,
                bandwidth_z=h,
                bandwidth_xz=h,
                warm_up_duration=100,
            )
            print(f"fitting response [{response}] of dataset [{dataset.name}]")
            model.fit(dataset)

            dataset_folder = Path(dataset.name.lower().replace(" ", "_"))
            dataset_folder.mkdir(parents=True, exist_ok=True)
            estimate_plot_title = f"estimate on {response}"
            estimate_plot_path = \
                    dataset_folder / f"estimate_{response}_h_{h}.pdf"
            plot_estimate(
                model,
                dataset,
                title=estimate_plot_title,
                path=estimate_plot_path,
            )
            # Make this plot only once
            if dataset.name == "poster dataset" and count == 0:
                density_plot_title = f"density estimate with h equal {h}"
                density_plot_path =  \
                        dataset_folder / f"density_estimate_h_{h}.pdf"
                plot_density_estimates(
                    model,
                    dataset,
                    title=density_plot_title,
                    path=density_plot_path
                )


@experiment("run all datasets")
def main():
    run_model_on_all_datasets()


if __name__ == "__main__":
    main()
