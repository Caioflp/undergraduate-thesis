""" Script to benchmark density ratio estimators.
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data.synthetic import make_benchmark_dataset
from src.models import KernelDensityRatio, DeepDensityRatio, AnalyticalDensityRatio
from src.scripts.utils import experiment


cm = 1/2.54
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8,
    "figure.figsize": (5*cm, 5*cm),
})


@experiment("benchmark-density-ratio", benchmark=True)
def benchmark_density_ratio(fit=False, plot=False):
    if fit:
        kernel_mse_list = []
        deep_mse_list = []
        for _ in tqdm(range(20)):
            data = make_benchmark_dataset(
                n_fit_samples=2600,
            )
            X_fit = data["X_fit"][:600]
            Z_fit = data["Z_fit"][:600]
            X_test = data["X_fit"][600:1600]
            Z_test = data["Z_fit"][1600:2600]
            model_kernel = KernelDensityRatio()
            model_deep = DeepDensityRatio()
            W_numerator_fit = np.concatenate([X_fit, Z_fit], axis=1)
            W_denominator_fit = np.concatenate([X_fit, np.roll(Z_fit, 2, axis=0)], axis=1)
            W_test = np.concatenate([X_test, np.roll(Z_test, 2, axis=0)], axis=1)
            model_kernel.fit(W_numerator_fit, W_denominator_fit)
            model_deep.fit(W_numerator_fit, W_denominator_fit)
            kernel_pred = model_kernel.predict(W_test)
            deep_pred = model_deep.predict(W_test)
            true_ratio = AnalyticalDensityRatio().predict(W_test)
            kernel_mse = np.mean((kernel_pred - true_ratio)**2)
            deep_mse = np.mean((deep_pred - true_ratio)**2)
            kernel_mse_list.append(kernel_mse)
            deep_mse_list.append(deep_mse)

        kernel_mse_array = np.array([np.log(mse)/np.log(10) for mse in kernel_mse_list])
        deep_mse_array = np.array([np.log(mse)/np.log(10) for mse in deep_mse_list])
        mse_arrays = {
            "uLSIF": kernel_mse_array,
            "Neural Net": deep_mse_array,
        }
        np.savez("mse_arrays.npz", **mse_arrays)
    else:
        mse_arrays = np.load("mse_arrays.npz")

    flierprops = dict(
        marker='o', markersize=3,
        linestyle='none', markeredgecolor='k',
    )
    fig, ax = plt.subplots()
    plot = ax.boxplot(
        mse_arrays.values(),
        labels=mse_arrays.keys(),
        patch_artist=True,
        flierprops=flierprops,
    )
    for line in plot['medians']:
        line.set(color="black")
    ax.set_title(r"$|| \hat{\Phi} - \Phi ||^2$")
    fig.text(0.5, -0.12, "Model", ha="center")
    fig.text(-0.1, 0.5, "Out of sample log-MSE", va="center", rotation="vertical")
    fig.autofmt_xdate()
    fig.savefig("density_ratio_mse.pdf", bbox_inches="tight")

    print(np.exp(np.log(10)*mse_arrays["uLSIF"]).mean())
    print(np.exp(np.log(10)*mse_arrays["uLSIF"]).std())

if __name__ == "__main__":
    benchmark_density_ratio(fit=False, plot=True)