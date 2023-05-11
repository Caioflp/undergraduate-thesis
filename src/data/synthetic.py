"""Synthetic data generation.

Author: @Caioflp

"""
from dataclasses import dataclass
import logging
from typing import Literal
from src.data.utils import InstrumentalVariableDataset

import numpy as np


LOGGER = logging.getLogger(__name__)


def make_low_dimensional_regression(
    n_samples: int = 1000, 
    response: Literal["sin", "step", "abs", "linear"] = "sin",
    seed: int = None,
) -> InstrumentalVariableDataset:
    """Creates a dataset for a low dimensional regression problem.

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
    dataset: InstrumentalVariableDataset
        Object containing the joint observations of X, Z and Y, as well as
        the denoised versions of Y.

    """
    LOGGER.info("Generating low dimensional dataset.")

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
    exact_Y = response_func(X)
    noisy_Y = exact_Y + eps + delta
    dataset = InstrumentalVariableDataset(X, Z, noisy_Y, exact_Y)
    return dataset

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = make_low_dimensional_regression(5000, response="abs")
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

