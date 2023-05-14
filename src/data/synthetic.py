"""Synthetic data generation.

Author: @Caioflp

"""
from dataclasses import dataclass
import logging
from typing import Literal
from src.data.utils import InstrumentalVariableDataset

import numpy as np
import scipy


LOGGER = logging.getLogger(__name__)


def make_poster_dataset(
    n_samples: int = 500,
    response: Literal["case_1", "case_2", "case_3"] = "case_1",
    sigma: float = 0.25,
    rho: float = 0.7,
    eta: float = 0.4,
    seed: int = None,
) -> InstrumentalVariableDataset:
    """Creates a dataset in which the instrument is uniformly distributed.

    Problem was taken from a research poster by Yuri Saporito, Yuri Resende and
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
    LOGGER.info("Generating poster dataset.")

    rng = np.random.default_rng(seed)
    norm_cdf = scipy.stats.norm.cdf

    response_dict = {
        "case_1": lambda x: np.power(x, 2) + 0.2*x,
        "case_2": lambda x: 2*np.power((x - 0.5)*(x - 0.5 > 0), 2) + 0.5*x,
        "case_3": lambda x: np.exp(np.abs(x - 0.5)),
    }
    assert response in ["case_1", "case_2", "case_3"]
    response_func = response_dict[response]

    W = rng.normal(loc=0, scale=1, size=(n_samples, 3))
    W_1, W_2, W_3 = W[:, 0], W[:, 1], W[:, 2]
    Z = norm_cdf(W_1)
    X = norm_cdf(rho*W_1 + np.sqrt(1 - np.power(rho, 2))*W_2)
    Y_denoised = response_func(X)
    eps = sigma*(eta*W_2 + np.sqrt(1 - np.power(eta, 2))*W_3)
    Y = Y_denoised + eps

    return InstrumentalVariableDataset(
        X, Z, Y, Y_denoised, "poster dataset"
    )
    

def make_deep_gmm_dataset(
    n_samples: int = 500, 
    response: Literal["sin", "step", "abs", "linear"] = "sin",
    seed: int = None,
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
    LOGGER.info("Generating deep gmm dataset.")

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
    return InstrumentalVariableDataset(
        X, Z, Y, Y_denoised, "deep gmm dataset"
    )

if __name__ == "__main__":
    import matplotlib.pyplot as plt

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

    dataset = make_poster_dataset(500, response="case_1")
    plt.scatter(
        dataset.X,
        dataset.Y,
        s=.2
    )
    plt.scatter(
        dataset.X,
        dataset.Y_denoised,
        s=.3
    )
    plt.show()
