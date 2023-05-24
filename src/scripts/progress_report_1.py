"""Script to compile everything that was done so far.

Date: 24/05/2023

Author: @Caioflp

"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KernelDensity as KDE
from sklearn.neighbors import KNeighborsRegressor as KNN

from src.data import InstrumentalVariableDataset
from src.data.synthetic import (
    make_poster_dataset,
    make_deep_gmm_dataset,
)
from src.models import FunctionalSGD
from src.scripts.utils import experiment


@experiment("progress_report")
def main():


if __name__ == "__main__":
    main()
