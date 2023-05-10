from dataclasses import dataclass

import numpy as np


@dataclass
class InstrumentalVariableDataset:
    X: np.ndarray
    Z: np.ndarray
    Y: np.ndarray
    Y_denoised: np.ndarray
