""" Script to create some graphs with examples of applications of 2SLS
regression

Author: @Caioflp

"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.models.utils import ensure_two_dimensional
from src.data import InstrumentalVariableDataset
from src.data.synthetic import (
    make_dummy_dataset,
)


# fig_path = Path("../thesis/fig/")
fig_path = Path("../presentation/fig/")

colors = mcolors.CSS4_COLORS
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
})


def compute_tsls_parameters(X, Z, Y):
    Xt_Z = X.T @ Z
    Zt_X = Z.T @ X
    Zt_Z = Z.T @ Z
    Zt_Y = Z.T @ Y
    beta = np.linalg.solve(
        Xt_Z @ np.linalg.solve(Zt_Z, Zt_X),
        Xt_Z @ np.linalg.solve(Zt_Z, Zt_Y)
    )
    return beta

def compute_ols_parameters(X, Y):
    return np.linalg.solve(X.T @ X, X.T @ Y)

if __name__ == "__main__":
    cm = 1/2.54
    # fig, axs = plt.subplots(
    #     nrows=2,
    #     ncols=1,
    #     figsize=(16*cm, 17*cm),
    #     layout="constrained",
    # )
    # noises = ["sensible", "absurd"]
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(16*cm, 8*cm),
        layout="constrained",
    )
    noises = ["absurd"]
    axs = [ax]
    for noise, ax in zip(noises, axs):
        dataset = make_dummy_dataset(
            n_samples=250,
            noise=noise,
            eta=0.6,
            rho=0.8,
            sigma=0.1,
        )
        X, Z, Y, Y_denoised = \
                dataset.X, dataset.Z, dataset.Y, dataset.Y_denoised
        X = ensure_two_dimensional(X)
        Z = ensure_two_dimensional(Z)
        Y = ensure_two_dimensional(Y)
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        Z = np.concatenate([np.ones((Z.shape[0], 1)), Z], axis=1)
        beta_tsls = compute_tsls_parameters(X, Z, Y)
        tsls_estimate = X @ beta_tsls
        beta_ols = compute_ols_parameters(X, Y)
        ols_estimate = X @ beta_ols

        # Plot data
        ax.scatter(
            X[:, 1].flatten(),
            Y.flatten(),
            c=colors["dimgrey"],
            s=5,
            alpha=.5,
            label=r"Samples from $(X, Y)$"
        )

        sorted_idx = np.argsort(X[:, 1].flatten())
        # Plot actual response
        ax.plot(
            X[sorted_idx, 1].flatten(),
            Y_denoised[sorted_idx].flatten(),
            c=colors["red"],
            linestyle="--",
            label=r"$h^{\star} (x) = 0.3 + 0.7x$"
        )

        # Plot TSLS estimate
        tsls_estimate_string = \
                fr"2SLS estimate: $y = {beta_tsls[0, 0]:.2f} + {beta_tsls[1, 0]:.2f}x$"
        ax.plot(
            X[sorted_idx, 1].flatten(),
            tsls_estimate[sorted_idx].flatten(),
            c=colors["blue"],
            label=tsls_estimate_string,
        )

        # Plot OLS estimate
        ols_estimate_string = \
                fr"OLS estimate: $y = {beta_ols[0, 0]:.2f} + {beta_ols[1, 0]:.2f}x$"
        ax.plot(
            X[sorted_idx, 1].flatten(),
            ols_estimate[sorted_idx].flatten(),
            c=colors["turquoise"],
            linestyle="-.",
            label=ols_estimate_string,
        )
        ax.legend()
        if noise == "sensible":
            ax.set_title(
                r"Mild codependence between $X$ and $\varepsilon$",
                fontsize=12,
            )
        elif noise == "absurd":
            ax.set_title(
                r"Strong codependence between $X$ and $\varepsilon$",
                fontsize=12,
            )
    # fig.suptitle(
    #     r"Two scenerios with endogeneity where $h^{\star}(x) = 0.3 + 0.7x$",
    #     # fontsize=14,
    # )
    fig.savefig(fig_path / "tsls_examples.pdf")

