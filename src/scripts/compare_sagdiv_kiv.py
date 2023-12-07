"""Script to compare SAGDIV with KIV

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
from src.models import FunctionalSGD, KIV
from src.scripts.utils import experiment

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
})

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
    model_sagdiv,
    model_kiv,
    dataset: InstrumentalVariableDataset,
    figsize: Tuple[int] = (7, 5),
    with_data = True,
    title: str = "SAGD-IV and KIV",
    ax = None,
) -> None:
    if ax is None:
        fig, ax = plt.subplots(layout="constrained", figsize=figsize)
    # Sorting is necessary to make line plots
    sorted_idx_observed = np.argsort(model_sagdiv.domain.observed_points.flatten())
    ax.plot(
        dataset.X.flatten()[sorted_idx_observed],
        dataset.Y_denoised[sorted_idx_observed],
        c="r",
        label=r"$h^{\star} (X)$",
        alpha=.8,
    )
    sort_idx = np.argsort(model_sagdiv.domain.all_points.flatten())
    sorted_x = model_sagdiv.domain.all_points.flatten()[sort_idx]
    sorted_estimate = model_sagdiv.estimate.on_all_points[sort_idx]
    sorted_last_estimate = \
            model_sagdiv.sequence_of_estimates.on_all_points[-1][sort_idx]
    ax.plot(
        sorted_x,
        sorted_estimate,
        c="b",
        label="SAGD-IV",
    )
    # ax.plot(
    #     sorted_x,
    #     sorted_last_estimate,
    #     c="k",
    #     label="Last estimate",
    # )
    ax.plot(
        sorted_x,
        model_kiv.predict(sorted_x),
        c="m",
        label="KIV"
    )
    if with_data:
        ax.scatter(
            dataset.X.flatten(),
            dataset.Y,
            c="k",
            s=1,
            alpha=0.2,
            label=r"$Y$",
        )

    ax.set_title(title)
    # ax.set_xlim(-4, 4)
    ax.legend()
    if ax is None:
        fig.savefig(title.lower().replace(" ", "_") + ".pdf")

# @experiment("new_version/sandbox")
@experiment("benchmark/benchmark_KIV")
def main():
    cm = 1/2.54
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(22*cm, 12*cm))
    fig.set_tight_layout(True)
    ax_sin = axs[0]
    ax_abs = axs[1]

    response = "sin"
    dataset_sagdiv, dataset_kiv = make_deep_gmm_dataset(
        n_samples=600, n_samples_only_z=1200, response=response,
        return_kiv_dataset=True,
    )
    model_kiv = KIV()
    model_kiv.fit(dataset_kiv)
    model_sagdiv = FunctionalSGD(
        lr="inv_n_samples",
        warm_up_duration=100,
        bound=10,
        nesterov=True,
    )
    model_sagdiv.fit(dataset_sagdiv)
    plot_estimate(
        model_sagdiv, model_kiv, dataset_sagdiv,
        title=r"$h^{\star} (x) = \sin (x)$", ax=ax_sin
    )

    response = "abs"
    dataset_sagdiv, dataset_kiv = make_deep_gmm_dataset(
        n_samples=600, n_samples_only_z=1200, response=response,
        return_kiv_dataset=True,
    )
    model_kiv = KIV()
    model_kiv.fit(dataset_kiv)
    model_sagdiv = FunctionalSGD(
        lr="inv_n_samples",
        warm_up_duration=100,
        bound=10,
        nesterov=False,
    )
    model_sagdiv.fit(dataset_sagdiv)
    plot_estimate(
        model_sagdiv, model_kiv, dataset_sagdiv,
        title=r"$h^{\star} (x) = |x|$", ax=ax_abs
    )

    fig.savefig("sagd-iv_kiv_comparison.pdf")

    
    # plt.hist(np.max(model.sequence_of_estimates.on_all_points, axis=0))
    # plt.show()
    # plot_estimate(model, dataset, title=f"Estimate for {response} in {dataset.name}")


if __name__ == "__main__":
    main()
