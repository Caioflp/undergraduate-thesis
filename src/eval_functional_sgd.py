"""Script to evaluate the FunctionalSGD algorithm

Author: Caio

"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.data import InstrumentalVariableDataset
from src.data.synthetic import (
    make_poster_dataset,
    make_deep_gmm_dataset,
)
from src.models import FunctionalSGD
from src.utils import experiment


def plot_data(
    dataset: InstrumentalVariableDataset,
    figsize: Tuple[int, int] = (7, 5),
    title : str = "Data"
) -> None:
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)
    ax.scatter(
        dataset.X.flatten(),
        dataset.Y,
        c="y",
        s=1.5,
        alpha=0.8,
        label="Observed response",
    )
    # Sorting is necessary to make line plots
    sort_idx = np.argsort(dataset.X.flatten())
    sorted_x = dataset.X[sort_idx].flatten()
    sorted_y_denoised = dataset.Y_denoised[sort_idx]
    ax.plot(
        sorted_x.flatten(),
        sorted_y_denoised,
        c="r",
        label="Denoised response",
    )
    ax.set_title(title)
    ax.legend()
    fig.savefig(title.lower() + ".pdf")
    
    
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


def plot_densities_for_poster_dataset(
    model: FunctionalSGD,
    dataset: InstrumentalVariableDataset,
    figsize: Tuple[int] = (7, 5),
    with_data = True,
    title: str = "Estimate",
) -> None:
    assert dataset.name == "poster dataset"
    fig, ax = plt.subplots(figsize=figsize)
    sort_idx = np.argsort(model.domain.all_points.flatten())
    sorted_x = model.domain.all_points[sort_idx]
    model_densities = np.exp(model.density_estimator_x.score_samples(sorted_x))
    ax.plot(
        sorted_x.flatten(),
        model_densities,
        c="b",
        label="Model estimate for p(x)"
    )


@experiment()
def main():
    dataset = make_poster_dataset(n_samples=100)
    model = FunctionalSGD()
    model.fit(dataset)
    print(model.density_estimator_x.sample(3))
    print(model.density_estimator_x == model.density_estimator_z ==
          model.density_estimator_xz)
    # plot_densities_for_poster_dataset(
    #     model, dataset, title="Estimate for case 2"
    # )

    # for response in ["sin", "step", "abs", "linear"]:
    #     print(f"Running for {response} data")
    #     dataset = make_deep_gmm_dataset(n_samples=500, response=response)
    #     model = FunctionalSGD()
    #     model.fit(dataset)

    #     # plot_data(dataset, title=f"Data_{response}")
    #     plot_estimate(model, dataset, title=f"Estimate_{response}")

    # for response in ["case_1", "case_2", "case_3"]:
    #     print(f"Running for {response} data")
    #     dataset = make_poster_dataset(n_samples=100, response=response)
    #     model = FunctionalSGD()
    #     model.fit(dataset)

    #     plot_estimate(model, dataset, title=f"Estimate_{response}")

    # print("Sequence of estimates:")
    # print(model.sequence_of_estimates.on_all_points[:3])
    # print(30*"-")
    # print("Final estimate:")
    # print(model.estimate.on_all_points)

    # dataset = make_poster_dataset(n_samples=100, response="case_2")
    # model = FunctionalSGD()
    # model.fit(dataset)
    # plot_estimate(model, dataset, title="Estimate for case 2")

    # print(np.all(model.domain.observed_points.flatten() == dataset.X))
    # sort_idx = np.argsort(model.domain.all_points.flatten())
    # print(np.all(
    #     (model.domain.all_points[sort_idx].flatten()
    #      == model.domain.all_points.flatten()[sort_idx])
    # ))

    # print("domain shapes:")
    # print("observed: ", model.domain.observed_points.shape)
    # print("grid: ", model.domain.grid_points.shape)
    # print("all: ", model.domain.all_points.shape)
    # 
    # print(50*"-")

    # print("estimates shapes")
    # print("observed: ", model.sequence_of_estimates.on_observed_points.shape)
    # print("grid: ", model.sequence_of_estimates.on_grid_points.shape)
    # print("all: ", model.sequence_of_estimates.on_all_points.shape)

    # print(50*"-")

    # print("final estimate shapes")
    # print("observed: ", model.estimate.on_observed_points.shape)
    # print("grid: ", model.estimate.on_grid_points.shape)
    # print("all: ", model.estimate.on_all_points.shape)


if __name__ == "__main__":
    main()
