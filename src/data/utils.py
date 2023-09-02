"""Data related utilities.

Author: Caioflp.

"""
from dataclasses import dataclass

import numpy as np


@dataclass
class InstrumentalVariableDataset:
    """Dataset for instrumental variable regression.

    """
    X: np.ndarray
    X_independent: np.ndarray
    Z: np.ndarray
    Z_independent: np.ndarray
    Z_loop: np.ndarray
    Y: np.ndarray
    Y_denoised: np.ndarray
    name: str = None
