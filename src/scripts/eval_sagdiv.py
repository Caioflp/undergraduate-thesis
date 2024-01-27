"""Script to evaluate the SAGDIV algorithm

Author: Caio

"""
import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data import SAGDIVDataset
from src.data.synthetic import (
    make_poster_dataset,
    make_deep_gmm_dataset,
)
from src.models import SAGDIV, DeepRegressionYZ, OperatorRegressionYZ
from src.scripts.utils import experiment, setup_logger


logger = logging.getLogger("src.experiment")


def plot_data(
    dataset: SAGDIVDataset,
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
    model: SAGDIV,
    dataset: SAGDIVDataset,
    figsize: Tuple[int] = (7, 5),
    with_data = True,
    title: str = "Estimate",
) -> None:
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)
    # Sorting is necessary to make line plots
    x_samples = dataset.X.flatten()
    sorted_idx = np.argsort(x_samples)
    sorted_x_samples = x_samples[sorted_idx]
    ax.plot(
        sorted_x_samples,
        dataset.Y_denoised[sorted_idx],
        c="r",
        label="Denoised response",
        alpha=.8,
    )
    x_linspace = np.linspace(np.min(x_samples), np.max(x_samples), 300)
    ax.plot(
        x_linspace,
        model.predict(x_linspace),
        c="b",
        label="Average estimate",
    )
    if with_data:
        ax.scatter(
            x_samples,
            dataset.Y,
            c="k",
            s=1,
            alpha=0.2,
            label="Observed response",
        )
    ax.set_title(title)
    # ax.set_xlim(-4, 4)
    ax.legend()
    fig.savefig(title.lower().replace(" ", "_") + ".pdf")


# @experiment("new_version/sandbox")
@experiment("eval-deep-YZ")
def main():
    response = "abs"
    n_samples = 600
    n_samples_only_z = 1200
    lr = "inv_n_samples"
    warm_up_duration = 100
    bound = 10
    message = f"""
                            Experiment data:
    response = {response}
    n_samples = {n_samples}
    n_samples_only_z = {n_samples_only_z}
    lr = {lr}
    warm_up_duration = {warm_up_duration}
    bound = {bound}
    """
    logger.info(message)
    dataset = make_deep_gmm_dataset(
        n_samples=n_samples,
        n_samples_only_z=n_samples_only_z,
        response=response,
    )
    # response = "case_2"
    # dataset = make_poster_dataset(n_samples=600, n_samples_only_z=2000,
    #                               response=response)
    mean_regressor_yz = DeepRegressionYZ(
        inner_layers_sizes=[128, 64, 32],
        activation="swish",
        batch_size=256,
        n_epochs=1000,
        learning_rate=0.01,
        weight_decay=0.01,
    )
    model = SAGDIV(lr=lr, warm_up_duration=warm_up_duration, bound=bound, mean_regressor_yz=mean_regressor_yz)
    model.fit(dataset)
    
    # plt.hist(np.max(model.sequence_of_estimates.on_all_points, axis=0))
    # plt.show()
    # plot_estimate(model, dataset, title=f"Estimate for {response} in {dataset.name}")
    plot_estimate(model, dataset, title="SAGD-IV")


if __name__ == "__main__":
    main()
