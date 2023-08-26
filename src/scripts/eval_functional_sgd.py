"""Script to evaluate the FunctionalSGD algorithm

Author: Caio

"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data import InstrumentalVariableDataset
from src.data.synthetic import (
    make_poster_dataset,
    make_deep_gmm_dataset,
)
from src.models import FunctionalSGD
from src.scripts.utils import experiment


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
    ax.set_xlim(-4, 4)
    ax.legend()
    fig.savefig(title.lower().replace(" ", "_") + ".pdf")


@experiment("new_version/eval_deep_gmm_dataset")
def main():
    dataset = make_deep_gmm_dataset(n_samples=500, n_samples_only_z=500, response="sin")
    model = FunctionalSGD()
    model.fit(dataset)
    plot_estimate(model, dataset)


if __name__ == "__main__":
    main()
