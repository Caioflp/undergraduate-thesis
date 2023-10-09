""" Script to create some graphs with examples of applications of 2SLS
regression

Author: @Caioflp

"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.models.utils import ensure_two_dimensional
from src.data import InstrumentalVariableDataset
from src.data.synthetic import (
    make_dummy_dataset,
)


colors = mcolors.CSS4_COLORS
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
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
    fig, axs = plt.subplots(nrows=1, ncols=2)
    responses = ["quadratic", "affine"]
    response_string = {
        "quadratic": r"$h^{\star} (x) = 0.3 + 0.2x + x^2$",
        "affine": r"$h^{\star} (x) = 0.3 + 0.7x$",
    }
    for response, ax in zip(responses, axs):
        dataset = make_dummy_dataset(
            n_samples=500,
            response=response,
            eta=0.4,
            rho=0.7,
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
            "r--",
            label=response_string[response]
        )

        # Plot TSLS estimate
        ax.plot(
            X[sorted_idx, 1].flatten(),
            tsls_estimate[sorted_idx].flatten(),
            c=colors["cornflowerblue"],
            label="TSLS fitted values"
        )

        # Plot OLS estimate
        ax.plot(
            X[sorted_idx, 1].flatten(),
            ols_estimate[sorted_idx].flatten(),
            c=colors["coral"],
            label="OLS fitted values"
        )
        ax.legend()
    plt.show()

