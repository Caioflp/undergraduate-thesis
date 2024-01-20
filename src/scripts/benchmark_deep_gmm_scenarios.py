"""Script to compare SAGDIV with KIV, DeepGMM and DeepIV

Author: @Caioflp

"""
import logging
from pathlib import Path
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import torch
from econml.iv.nnet import DeepIV

from src.data.synthetic import make_benchmark_dataset
from src.models import SAGDIV, KIV
from src.scripts.utils import experiment
from src.DeepGMM.scenarios.abstract_scenario import AbstractScenario
from src.DeepGMM.methods.toy_model_selection_method import ToyModelSelectionMethod as DeepGMM


logger = logging.getLogger("src.scripts.benchmarks")


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
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
        trian_x = train_x.cuda()
        trian_z = train_z.cuda()
        trian_y = train_y.cuda()
        val_x = val_x.cuda()
        val_z = val_z.cuda()
        val_y = val_y.cuda()

    method = DeepGMM(enable_cuda=enable_cuda)
    method.fit(train_x, train_z, train_y, val_x, val_z, val_y, verbose=True)
    h_hat_test = method.predict(test_x).detach().numpy()
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_sagd_iv(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
):
    pass


def train_eval_store_deep_iv(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
):
    """ DeepIV evaluation function.

        The EconML package does not specify how to split the data, so we pass all
        samples to the fit method.

        `context` here means DeepIV's X variable, while our X variable is DeepIV's p variable
    """
    n_samples = n_rv_samples // 3

    train_x = data["X_fit"][:n_samples] 
    train_z = data["Z_fit"][:n_samples] 
    train_y = data["Y_fit"][:n_samples] 

    test_x = data["X_test"]


def train_eval_store_kiv(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
):
    pass


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
    model_name_list = ["DeepGMM"]#, "KIV", "DeepIV", "SAGD-IV"]
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


def plot_MSEs():
    pass


def plot_graphs():
    pass

@experiment("benchmarks/", benchmark=True)
def main():
    eval_models_accross_scenarios()


if __name__ == "__main__":
    main()
