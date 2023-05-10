"""Entry point to run stuff.

Author: @Caioflp

"""
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neighbors import KernelDensity as KDE

from src.data.synthetic import make_low_dimensional_regression
from src.models import FunctionalGD, FunctionalSGD


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    model.fit(dataset)
    print(os.getcwd())
    model.make_plots(dataset)





if __name__ == "__main__":
    main()
