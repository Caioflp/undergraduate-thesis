"""Script to evaluate the SAGDIV algorithm on binary response data

Author: Caio

"""
import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data import InstrumentalVariableDataset
from src.data.synthetic import make_binary_response_dataset
from src.models import LogisticRegressionYZ
from src.models import SAGDIV
from src.models.utils import BCELogisticLoss, QuadraticLoss
from src.scripts.utils import experiment, setup_logger


logger = logging.getLogger("src.experiment")


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
    model: SAGDIV,
    dataset: InstrumentalVariableDataset,
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
@experiment("binary_response")
def main():
    response = "linear"
    scale = np.sqrt(0.1)
    n_samples = 600
    n_samples_only_z = 1200
    lr = 0.1
    initial_value = 0
    warm_up_duration = 100
    bound = 10
    message = f"""
                            Experiment data:
    response = {response}
    scale = {scale}
    n_samples = {n_samples}
    n_samples_only_z = {n_samples_only_z}
    lr = {lr}
    initial_value = {initial_value}
    warm_up_duration = {warm_up_duration}
    bound = {bound}
    """
    logger.info(message)
    dataset = make_binary_response_dataset(
        n_samples=n_samples,
        n_samples_only_z=n_samples_only_z,
        response=response,
        scale=scale,
    )
    # response = "case_2"
    # dataset = make_poster_dataset(n_samples=600, n_samples_only_z=2000,
    #                               response=response)
    model = SAGDIV(
        lr=lr,
        loss=BCELogisticLoss(scale=scale),
        mean_regressor_yz=LogisticRegressionYZ(),
        initial_value=initial_value,
        warm_up_duration=warm_up_duration,
        bound=bound,
    )
    model.fit(dataset)
    
    # plt.hist(np.max(model.sequence_of_estimates.on_all_points, axis=0))
    # plt.show()
    # plot_estimate(model, dataset, title=f"Estimate for {response} in {dataset.name}")
    plot_estimate(model, dataset, title="SAGD-IV")


if __name__ == "__main__":
    main()
