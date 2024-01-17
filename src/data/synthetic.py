"""Synthetic data generation.

Author: @Caioflp

"""
from dataclasses import dataclass
import logging
from typing import Literal
from src.data.utils import InstrumentalVariableDataset, KIVDataset

import numpy as np
import scipy


logger = logging.getLogger(__name__)


def make_dummy_dataset(
    n_samples: int = 500,
    n_samples_only_z: int = 500,
    noise: Literal["sensible", "absurd"] = "sensible",
    sigma: float = 0.25,
    rho: float = 0.7,
    eta: float = 0.4,
    seed: int = None,
) -> InstrumentalVariableDataset:
    """Creates a dataset in which the instrument is uniformly distributed.

    Problem was taken from a research poster by Yuri Saporito, Yuri Rezende and
    Rodrigo Targino.

    Parameters
    ----------
    n_samples: int
        Number of samples on the dataset. Each sample is a joint
        observation of X, Z and Y.
    response: str
        Specifies the response function for the data generation process.
    seed: int
        Seed for RNG.

    Returns
    -------
    InstrumentalVariableDataset
        Object containing the joint observations of X, Z and Y, as well as
        the denoised versions of Y.

    """
    logger.info("Generating poster dataset.")

    rng = np.random.default_rng(seed)
    norm_cdf = scipy.stats.norm.cdf

    assert noise in ["sensible", "absurd"]
    response_func = lambda x: .3 + .7*x

    W = rng.normal(loc=0, scale=1, size=(n_samples, 3))
    W_1, W_2, W_3 = W[:, 0], W[:, 1], W[:, 2]
    Z = norm_cdf(W_1)
    X = norm_cdf(rho*W_1 + np.sqrt(1 - np.power(rho, 2))*W_2)
    Y_denoised = response_func(X)
    if noise == "sensible":
        eps = sigma*(eta*W_2 + np.sqrt(1 - np.power(eta, 2))*W_3)
    elif noise == "absurd":
        eps = (
            sigma*(eta*W_2 + np.sqrt(1 - np.power(eta, 2))*W_3)
            + 6*sigma*(W_2 - 0.7)*(W_2 > 0.7)
            + 2*sigma*(W_2 - 0.3)*(W_2 < 0.3)
        )

    Y = Y_denoised + eps

    Z_loop = norm_cdf(rng.normal(loc=0, scale=1, size=n_samples_only_z))

    return InstrumentalVariableDataset(
        X, Z, Z_loop, Y, Y_denoised, "poster dataset"
    )


def make_poster_dataset(
    n_samples: int = 500,
    n_samples_only_z: int = 500,
    response: Literal["case_1", "case_2", "case_3", "affine", "sin"] = "case_1",
    sigma: float = 0.25,
    rho: float = 0.7,
    eta: float = 0.4,
    seed: int = None,
) -> InstrumentalVariableDataset:
    """Creates a dataset in which the instrument is uniformly distributed.

    Problem was taken from a research poster by Yuri Saporito, Yuri Rezende and
    Rodrigo Targino.

    Parameters
    ----------
    n_samples: int
        Number of samples on the dataset. Each sample is a joint
        observation of X, Z and Y.
    response: str
        Specifies the response function for the data generation process.
    seed: int
        Seed for RNG.

    Returns
    -------
    InstrumentalVariableDataset
        Object containing the joint observations of X, Z and Y, as well as
        the denoised versions of Y.

    """
    logger.info("Generating poster dataset.")

    rng = np.random.default_rng(seed)
    norm_cdf = scipy.stats.norm.cdf

    response_dict = {
        "case_1": lambda x: np.power(x, 2) + 0.2*x,
        "case_2": lambda x: 2*np.power((x - 0.5)*(x - 0.5 > 0), 2) + 0.5*x,
        "case_3": lambda x: np.exp(np.abs(x - 0.5)),
        "affine": lambda x: .3 + .7*x,
        "quadratic": lambda x: .3 + .2*x + x**2,
    }
    assert response in ["case_1", "case_2", "case_3", "affine", "quadratic"]
    response_func = response_dict[response]

    W = rng.normal(loc=0, scale=1, size=(n_samples, 3))
    W_1, W_2, W_3 = W[:, 0], W[:, 1], W[:, 2]
    Z = norm_cdf(W_1)
    X = norm_cdf(rho*W_1 + np.sqrt(1 - np.power(rho, 2))*W_2)
    Y_denoised = response_func(X)
    eps = sigma*(eta*W_2 + np.sqrt(1 - np.power(eta, 2))*W_3)
    Y = Y_denoised + eps

    Z_loop = norm_cdf(rng.normal(loc=0, scale=1, size=n_samples_only_z))

    return InstrumentalVariableDataset(
        X, Z, Z_loop, Y, Y_denoised, "poster dataset"
    )
    

def make_deep_gmm_dataset(
    n_samples: int = 500, 
    n_samples_only_z: int = 500,
    response: Literal["sin", "step", "abs", "linear"] = "sin",
    seed: int = None,
    return_kiv_dataset: bool = False,
) -> InstrumentalVariableDataset:
    """Creates a dataset in which the instrument is uniformly distributed.

    Problem was taken from arXiv:1905.12495v2.

    Parameters
    ----------
    n_samples: int
        Number of samples on the dataset. Each sample is a joint
        observation of X, Z and Y.
    response: str
        Specifies the response function for the data generation process.
    seed: int
        Seed for RNG.

    Returns
    -------
    InstrumentalVariableDataset
        Object containing the joint observations of X, Z and Y, as well as
        the denoised versions of Y.

    """

    response_dict = {
        "sin": np.sin,
        "step": lambda x: x >= 0,
        "abs": np.abs,
        "linear": lambda x: x,
    }
    assert response in ["sin", "step", "abs", "linear"]
    response_func = response_dict[response]

    rng = np.random.default_rng(seed)

    Z = rng.uniform(low=-3, high=3, size=(n_samples, 2))
    eps = rng.normal(loc=0, scale=1, size=n_samples)
    gamma = rng.normal(loc=0, scale=np.sqrt(0.1), size=n_samples)
    delta = rng.normal(loc=0, scale=np.sqrt(0.1), size=n_samples)
    X = Z[:, 0] + eps + gamma
    Y_denoised = response_func(X)
    Y = Y_denoised + eps + delta

    
    Z_loop = rng.uniform(low=-3, high=3, size=(n_samples_only_z, 2))
    eps_loop = rng.normal(loc=0, scale=1, size=n_samples_only_z)
    gamma_loop = rng.normal(loc=0, scale=np.sqrt(0.1), size=n_samples_only_z)
    delta_loop = rng.normal(loc=0, scale=np.sqrt(0.1), size=n_samples_only_z)
    X_loop = Z_loop[:, 0] + eps_loop + gamma_loop
    Y_loop = response_func(X_loop) + eps_loop + delta_loop

    sagdiv_dataset = InstrumentalVariableDataset(
            X, Z, Z_loop, Y, Y_denoised, "deep gmm dataset"
        )
    logger.info("Deep GMM dataset generated.")
    if not return_kiv_dataset:
        return sagdiv_dataset
    else: 
        n_samples_tilde = Z_loop.shape[0]//2
        Z_tilde = Z_loop[:n_samples_tilde]
        Y_tilde = Y_loop[:n_samples_tilde]
        kiv_dataset = KIVDataset(X, Z, Y, Z_tilde, Y_tilde, "deep gmm dataset")
        # KIV needs one dataset of (X, Y, Z) and one of (X, Z).
        # We return the same (X, Y, Z) dataset that we use in SAGDIV
        # and return a dataset of (Z_tilde, Y_tilde) samples,
        # where the Z_tilde samples are the same we use in the SAGDIV
        # training loop.
        # To be more precise, if we use M samples of Z in the SAGDIV loop,
        # We use M/2 pairs of (Z_tilde, Y_tilde) for the second stage
        # of KIV, so that the total number of random variables used by
        # each method is the same.
        return sagdiv_dataset, kiv_dataset



def make_binary_response_dataset(
    n_samples: int = 500, 
    n_samples_only_z: int = 500,
    response: Literal["sin", "linear"] = "sin",
    seed: int = None,
) -> InstrumentalVariableDataset:
    """Creates a dataset in which the response is binary

    Parameters
    ----------
    n_samples: int
        Number of samples on the dataset. Each sample is a joint
        observation of X, Z and Y.
    response: str
        Specifies the response function for the data generation process.
    seed: int
        Seed for RNG.

    Returns
    -------
    InstrumentalVariableDataset
        Object containing the joint observations of X, Z and Y, as well as
        the denoised versions of Y.

    """

    response_dict = {
        "sin": np.sin,
        "step": lambda x: x >= 0,
        "abs": np.abs,
        "linear": lambda x: x,
    }
    # Computes the conditional expectation of response_func(X) given Z = z
    conditional_mean_dict = {
        "sin": lambda Z: np.sin(Z[:, 0])*np.pi*np.exp(-(0.1**2)/2)/np.sinh(np.pi),
        "linear": lambda Z: Z[:, 0]
    }
    assert response in ["sin", "linear"]
    response_func = response_dict[response]
    conditional_mean_given = conditional_mean_dict[response]

    rng = np.random.default_rng(seed)

    Z = rng.uniform(low=-3, high=3, size=(n_samples, 2))
    eta = rng.logistic(loc=0, scale=1, size=n_samples)
    gamma = rng.normal(loc=0, scale=0.1, size=n_samples)
    X = Z[:, 0] + eta + gamma
    Y_denoised = response_func(X)
    Y = (conditional_mean_given(Z) + eta > 0).astype(float)

    
    Z_loop = rng.uniform(low=-3, high=3, size=(n_samples_only_z, 2))
    sagdiv_dataset = InstrumentalVariableDataset(
            X, Z, Z_loop, Y, Y_denoised, "Binary response dataset"
        )
    logger.info("Binary response dataset generated.")
    return sagdiv_dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = make_binary_response_dataset(500, response="sin")
    # plt.scatter(
    #     dataset.X,
    #     dataset.Y,
    #     s=.2
    # )
    # plt.scatter(
    #     dataset.X,
    #     dataset.Y_denoised,
    #     s=.3
    # )
    plt.scatter(
        dataset.Z[:, 0],
        dataset.Y_denoised,
        s = 1,
    )
    plt.scatter(
        dataset.Z[:, 0],
        dataset.Y,
        s = 1,
    )
    plt.show()

    # dataset = make_deep_gmm_dataset(500, response="abs")
    # plt.scatter(
    #     dataset.X,
    #     dataset.Y,
    #     s=.2
    # )
    # plt.scatter(
    #     dataset.X,
    #     dataset.Y_denoised,
    #     s=.3
    # )
    # plt.show()

    # dataset = make_poster_dataset(500, response="case_1")
    # plt.scatter(
    #     dataset.X,
    #     dataset.Y,
    #     s=.2
    # )
    # plt.scatter(
    #     dataset.X,
    #     dataset.Y_denoised,
    #     s=.3
    # )
    # plt.show()
