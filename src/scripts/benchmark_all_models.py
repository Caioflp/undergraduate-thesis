"""Script to compare SAGDIV with KIV, DeepGMM and DeepIV

Author: @Caioflp

"""
import logging
from pathlib import Path
from typing import Tuple, Dict

import econml
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import torch
from econml.iv.nnet import DeepIV
from tensorflow import keras

from src.data.synthetic import make_benchmark_dataset
from src.data.utils import SAGDIVDataset, KIVDataset
from src.DeepGMM.scenarios.abstract_scenario import AbstractScenario
from src.DeepGMM.methods.toy_model_selection_method import ToyModelSelectionMethod as DeepGMM
from src.models import SAGDIV, KIV
from src.scripts.utils import experiment


logger = logging.getLogger("src.scripts.benchmarks")
COLOR_PER_MODEL = {
    "DeepGMM": "pink",
    "KIV": "orange",
    "DeepIV": "violet",
    "SAGD-IV": "lightblue",
}


cm = 1/2.54
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "figure.figsize": (20*cm, 10*cm)
})


def train_eval_store_deep_gmm(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
):
    """ DeepGMM evaluation function.

        We adopt a 50/50% train/validation split, as in the original article.
    """
    enable_cuda = torch.cuda.is_available()
    n_samples = n_rv_samples // 3 // 2

    train_x = torch.as_tensor(data["X_fit"][:n_samples]).double()
    train_z = torch.as_tensor(data["Z_fit"][:n_samples]).double()
    train_y = torch.as_tensor(data["Y_fit"][:n_samples]).double()

    test_x = torch.as_tensor(data["X_test"]).double()

    val_x = torch.as_tensor(data["X_fit"][n_samples:2*n_samples]).double()
    val_z = torch.as_tensor(data["Z_fit"][n_samples:2*n_samples]).double()
    val_y = torch.as_tensor(data["Y_fit"][n_samples:2*n_samples]).double()

    if enable_cuda:
        train_x = train_x.cuda()
        train_z = train_z.cuda()
        train_y = train_y.cuda()
        val_x = val_x.cuda()
        val_z = val_z.cuda()
        val_y = val_y.cuda()
        test_x = test_x.cuda()

    method = DeepGMM(enable_cuda=enable_cuda)
    method.fit(train_x, train_z, train_y, val_x, val_z, val_y, verbose=True)
    if enable_cuda:
        h_hat_test = method.predict(test_x).cpu().detach().numpy()
    else:
        h_hat_test = method.predict(test_x).detach().numpy()
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_sagd_iv(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
):
    """ SAGD-IV evaluation function.

        Let N denote the number of triplets (X, Y, Z) that will be used to fit r, Phi and P.
        We want the number of Z samples used in the loop to be 2*N.
        Hence, we have
            n_rv_samples = 3*N + 2*N = 5*N
        and N = n_rv_samples // 5

    """

    n_samples = n_rv_samples // 5
    train_x = data["X_fit"][:n_samples] 
    train_z = data["Z_fit"][:n_samples] 
    train_y = data["Y_fit"][:n_samples] 
    test_x = data["X_test"]

    train_loop_z = data["Z_fit"][n_samples:n_samples + 2*n_samples]

    model = SAGDIV(lr="inv_n_samples", warm_up_duration=100, bound=10)
    model.fit(SAGDIVDataset(train_x, train_z, train_loop_z, train_y))
    h_hat_test = model.predict(test_x)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_deep_iv(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
):
    """ DeepIV evaluation function.

        The EconML package does not specify how to split the data, so we pass all
        samples to the fit method.

        `context` here means DeepIV's X variable, while our X variable is DeepIV's p variable.
        DeepIV requires a context variable, but this DGP does not have one.
        Hence, we will set a context variable to be constant and equal to 0.
    """
    n_samples = n_rv_samples // 3

    train_x = data["X_fit"][:n_samples] 
    train_z = data["Z_fit"][:n_samples] 
    train_y = data["Y_fit"][:n_samples] 

    train_context = np.zeros((train_x.shape[0], 1), dtype=float)

    test_x = data["X_test"]

    dim_x = train_x.shape[1]
    dim_z = train_z.shape[1]
    dim_context = train_context.shape[1]

    treatment_model = keras.Sequential([
            # keras.layers.Dense(64, activation='tanh', input_shape=(dim_z+dim_context,)),
            # keras.layers.Dropout(0.17),
            keras.layers.Dense(32, activation='tanh'),
            keras.layers.Dropout(0.17),
            keras.layers.Dense(16, activation='tanh'),
            keras.layers.Dropout(0.17)
    ])

    response_model = keras.Sequential([
            # keras.layers.Dense(64, activation='relu', input_shape=(dim_x+dim_context,)),
            # keras.layers.Dropout(0.17),
            keras.layers.Dense(32, activation='tanh'),
            keras.layers.Dropout(0.17),
            # keras.layers.Dense(16, activation='tanh'),
            # keras.layers.Dropout(0.17),
            keras.layers.Dense(1),
    ])

    keras_fit_options = {
        "epochs": 30,
        "validation_split": 0.1,
        "callbacks": [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
    }
    model = DeepIV(
        n_components=10,
        m=lambda z, context : treatment_model(keras.layers.concatenate([z, context])),
        h=lambda x, context: response_model(keras.layers.concatenate([x, context])),
        n_samples=1,
        use_upper_bound_loss=False,
        n_gradient_samples=1,
        optimizer="adam",
        first_stage_options=keras_fit_options,
        second_stage_options=keras_fit_options,
    )
    
    # Monkey patching.
    # As of writing this code, this function on the econml.iv.nnet._deepiv module
    # needs to be updated because it calls keras.backend.logsumexp, which no longer exists.
    def fixed_mog_loss_model(n_components, d_t):
        pi = keras.layers.Input((n_components,))
        mu = keras.layers.Input((n_components, d_t))
        sig = keras.layers.Input((n_components,))
        t = keras.layers.Input((d_t,))
        # || t - mu_i || ^2
        d2 = keras.layers.Lambda(lambda d: keras.backend.sum(keras.backend.square(d), axis=-1),
                    output_shape=(n_components,))(
            keras.layers.Subtract()([keras.layers.RepeatVector(n_components)(t), mu])
        )
        # keras.layerskeras.layers = C - log(sum(pi_i/sig^d * exp(-d2/(2*sig^2))))
        # Use logsumexp for numeric stability:
        # keras.layerskeras.layers = C - log(sum(exp(-d2/(2*sig^2) + log(pi_i/sig^d))))
        # TODO: does the numeric stability actually make any difference?
        def make_logloss(d2, sig, pi):
            # return -keras.backend.logsumexp(-d2 / (2 * keras.backend.square(sig)) + keras.backend.log(pi / keras.backend.pow(sig, d_t)), axis=-1)
            return -tf.math.reduce_logsumexp(-d2 / (2 * keras.backend.square(sig)) + keras.backend.log(pi / keras.backend.pow(sig, d_t)), axis=-1)
        ll = keras.layers.Lambda(lambda dsp: make_logloss(*dsp), output_shape=(1,))([d2, sig, pi])
        m = keras.Model([pi, mu, sig, t], [ll])
        return m
    econml.iv.nnet._deepiv.mog_loss_model = fixed_mog_loss_model

    model.fit(Y=train_y, T=train_x, X=train_context, Z=train_z)
    test_context = np.zeros((test_x.shape[0], 1), dtype=float)
    h_hat_test = model.predict(test_x, test_context)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_kiv(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
):
    """ KIV evaluation function.

        KIV needs one dataset of (X, Y, Z) and one of (Z, Y).
        We make each dataset have the same amount of tuple samples, that is,
        letting N denote that number, one dataset is comprised of N samples from the triplet
        (X, Y, Z) and the other is comprised of N samples from the pair (Z, Y).
        
        So we must have
            n_rv_samples = 3*N + 2*N = 5*N
        and N = n_rv_samples // 5

    """
    n_samples = n_rv_samples // 5
    train_x = data["X_fit"][:n_samples] 
    train_z = data["Z_fit"][:n_samples] 
    train_y = data["Y_fit"][:n_samples] 
    train_z_tilde = data["Z_fit"][n_samples:2*n_samples]
    train_y_tilde = data["Y_fit"][n_samples:2*n_samples]

    test_x = data["X_test"]

    model = KIV()
    model.fit(KIVDataset(train_x, train_z, train_y, train_z_tilde, train_y_tilde))
    h_hat_test = model.predict(test_x)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store(model_name: str, *args):
    model_eval_function_dict = {
        "DeepGMM": train_eval_store_deep_gmm,
        "SAGD-IV": train_eval_store_sagd_iv,
        "KIV": train_eval_store_kiv,
        "DeepIV": train_eval_store_deep_iv,
    }
    return model_eval_function_dict[model_name](*args)


def eval_models_accross_scenarios(
    scenarios: list = ["step", "sin", "abs", "linear"],
    model_name_list: list = ["DeepGMM", "KIV", "DeepIV", "SAGD-IV"],
    n_runs: int = 20,
    n_triplet_samples: int = 5000,
    n_rv_samples_for_fit: int = 3000,
    n_test_samples: int = 1000
):
    """ Evaluates each model `n_runs` times in each scenario.

    Since different algorithms require different proportions of X, Z and Y samples,
    our experiment assumes that each model has access to the same total amount of random
    variable samples, but those might be distributed differently accross models.
    For example, if all models should be fitted using 3000 r.v. variables, then DeepGMM
    may use 800 (X, Y, Z) samples for training and another 200 (X, Y, Z) samples for validation,
    while SAGD-IV might use 600 (X, Y, Z) samples for fitting the preliminary r, P and Phi
    estimators and another 1200 Z samples for the SAGD loop.

    Parameters
    ----------
    scenarios: list
        All scenarios which should be evaluated.
    n_runs: int
        Number times to evaluate each model in each scenario.
    n_triplet_samples: int
        Amount of (X, Y, Z) triplet samples to draw. It might be the case that not all of them
        will be actually used. We just want to make sure every algorithm gets what it needs.
    n_rv_samples_for_fitting: int
        Amount of random variable samples each model is allowed to use during the fitting process.
    n_test_samples: int
        Number of X samples in which the models will be evaluated.


    Experiment structure:

    for each scenario:
        for each run:
            generate and save new data
            for each method:
                train method
                store predictions on x_test
                compute test mse
        store test mse data accross runs for each model

    """
    model_mse_dict = {name: np.empty(n_runs, dtype=float) for name in model_name_list}
    for scenario in scenarios:
        scenario_dir = Path(scenario)
        scenario_dir.mkdir(exist_ok=True)
        logger.info(f"Entering scenario: {scenario.upper()}")
        for run_number in range(n_runs):
            logger.info(f"Starting run number {run_number}")
            run_dir = scenario_dir / ("run_" + str(run_number))
            run_dir.mkdir(exist_ok=True)
            data = make_benchmark_dataset(
                n_triplet_samples,
                n_test_samples,
                scenario,
            )
            logger.info(f"Generated {scenario.upper()} scenario benchmark data.")
            np.savez(run_dir / "data.npz", **data)
            for model_name in model_name_list:
                model_file = run_dir / (model_name.lower() + ".npz")
                h_hat = train_eval_store(model_name, data, n_rv_samples_for_fit, model_file)
                mse = np.mean(np.square(data["h_star_test"] - h_hat))
                model_mse_dict[model_name][run_number] = mse
            logger.info("Evaluated all models")
        report = f"MSE results for scenario {scenario.upper()}:\n"
        for model_name in model_name_list:
            mse_array = model_mse_dict[model_name]
            mean, std = np.mean(mse_array), np.std(mse_array)
            report += f"{model_name}: {mean:1.2e} ± {std:1.2e}\n" 
        logger.info(report)
        np.savez(scenario_dir / "mse_arrays.npz", **model_mse_dict)


def plot_MSEs(
    scenarios: list = ["step", "sin", "abs", "linear"],
    model_name_list: list = ["DeepGMM", "KIV", "DeepIV", "SAGD-IV"],
):
    n_scenarios = len(scenarios)
    fig, axs = plt.subplots(1, n_scenarios, sharey=True)
    for i, scenario in enumerate(scenarios):
        scenario_dir = Path(scenario)
        mse_arrays = np.load(scenario_dir / "mse_arrays.npz")
        mse_arrays = {k: np.log(mse_arrays[k])/np.log(10) for k in mse_arrays}
        plot = axs[i].boxplot(mse_arrays.values(), labels=mse_arrays.keys(), patch_artist=True)
        for patch, model_name in zip(plot['boxes'], model_name_list):
            patch.set(facecolor=COLOR_PER_MODEL[model_name])
        for line, model_name in zip(plot['medians'], model_name_list):
            line.set(color="black")
        axs[i].set_title(scenario.title())
    # fig.tight_layout()
    fig.text(0.5, 0.02, "Model", ha="center")
    fig.text(0.03, 0.5, "Out of sample log-MSE", va="center", rotation="vertical")
    fig.autofmt_xdate()
    fig.savefig("mse.pdf")


def plot_graphs(
    n_runs: int = 20,
    scenarios: list = ["step", "sin", "abs", "linear"],
    model_name_list: list = ["DeepGMM", "KIV", "DeepIV", "SAGD-IV"],
):
    n_scenarios = len(scenarios)
    n_models = len(model_name_list)
    # Choose a random run
    random_run = np.random.choice(n_runs)
    # random_run = 5

    fig, axs = plt.subplots(
        n_scenarios,
        n_models+1,
        sharey="row",
        sharex=True,
        figsize=(20*cm, 13*cm)
        )
    # fig.tight_layout()
    for i, scenario in enumerate(scenarios):
        run_dir = Path(scenario + "/" + f"run_{random_run}")
        data_file = run_dir / "data.npz"
        data = np.load(data_file)
        X = data["X_fit"].flatten()
        Y = data["Y_fit"]
        # Plot the data
        axs[i, 0].scatter(
            X, Y, c="green", s=1.5, alpha=0.8,
        )
        # Sorting is necessary to make line plots
        sort_idx = np.argsort(X)
        sorted_x = X[sort_idx]
        sorted_h_star = data["h_star_fit"][sort_idx]
        axs[i, 0].plot(
            sorted_x,
            sorted_h_star,
            c="k",
        )
        for j, model_name in enumerate(model_name_list, start=1):
            # Plot model estimates
            axs[i, j].plot(
                sorted_x,
                sorted_h_star,
                c="k",
            )
            X_test = data["X_test"].flatten()
            sorted_idx = np.argsort(X_test)
            sorted_x_test = X_test[sorted_idx]
            model_file = run_dir / (model_name.lower() + ".npz")
            h_hat_test = np.load(model_file)["h_hat_test"]
            sorted_h_hat = h_hat_test[sorted_idx]
            axs[i, j].plot(
                sorted_x_test,
                sorted_h_hat,
                c="b",
            )
    cols = ["Data"] + model_name_list
    rows = [scenario.title() for scenario in scenarios]
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(row)#, rotation=0)#, size='large')
    fig.savefig("graph_plots.pdf")

@experiment("benchmarks/simpler-neural-nets-architectures", benchmark=True)
def main():
    n_runs = 20
    eval_models_accross_scenarios(
        scenarios=["step", "abs", "linear", "sin"],
        n_runs=n_runs,
        n_triplet_samples=5000,
        n_rv_samples_for_fit=3000,
        n_test_samples=1000,
    )
    plot_MSEs()
    plot_graphs(n_runs=n_runs)


if __name__ == "__main__":
    main()
