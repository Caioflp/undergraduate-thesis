"""Script to evaluate the FunctionalSGD algorithm

Author: Caio

"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.data import InstrumentalVariableDataset
from src.data.synthetic import make_poster_dataset
from src.models import FunctionalSGD
from src.utils import experiment


def plot_data(
    dataset: InstrumentalVariableDataset,
    figsize: Tuple[int, int] = (5, 5),
) -> None:
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)
    title = "Data"
    ax.scatter(
        dataset.X.flatten(),
        dataset.Y,
        c="y",
        s=1.5,
        alpha=0.8,
        label="Observed response",
    )
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
    figsize: Tuple[int] = (5, 5),
) -> None:
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)
    title = "Estimate"
    sort_idx = np.argsort(model.domain.all_points.flatten())
    sorted_x = model.domain.all_points[sort_idx].flatten()
    sorted_estimate = model.estimate.on_all_points[sort_idx]
    sorted_last_estimate = \
            model.sequence_of_estimates.on_all_points[-1][sort_idx]
    ax.scatter(
        dataset.X.flatten(),
        dataset.Y_denoised,
        c="r",
        label="Denoised response",
        s=3,
        alpha=.8,
    )
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
    ax.set_title(title)
    ax.legend()
    fig.savefig(title.lower().replace(" ", "_") + ".pdf")


@experiment
def main():
    dataset = make_poster_dataset(n_samples=100)
    model = FunctionalSGD()
    model.fit(dataset)

    plot_data(dataset)
    plot_estimate(model, dataset)

if __name__ == "__main__":
    main()
