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
    Z: np.ndarray
    Y: np.ndarray
    Y_denoised: np.ndarray
