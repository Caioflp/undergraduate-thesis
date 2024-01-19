"""Data related utilities.

Author: Caioflp.

"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SAGDIVDataset:
    """Dataset for instrumental variable regression.

    """
    X: np.ndarray
    Z: np.ndarray
    Z_loop: np.ndarray
    Y: np.ndarray
    Y_denoised: np.ndarray
    name: str = None

@dataclass
class KIVDataset:
    """Dataset for instrumental variable regression using KIV

    """
    X: np.ndarray
    Z: np.ndarray
    Y: np.ndarray
    Z_tilde: np.ndarray
    Y_tilde: np.ndarray
    name: str = None
