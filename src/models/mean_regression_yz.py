""" This module implements estimators of E[Y|Z]
"""
import abc
import logging

import numpy as np

from src.models import ConditionalMeanOperator


logger = logging.getLogger("src.models.mean_regression_yz")


class MeanRegressionYZ(abc.ABC):
    def __init__(self):
        self.is_fitted = False

    @abc.abstractmethod
    def fit(
        self,
        Z: np.ndarray,
        Z_loop: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        pass

    @abc.abstractmethod
    def predict(
        self,
        z: np.ndarray,
        Y: np.ndarray,
        i: int,
        for_loop: bool = False,
    ) -> np.ndarray:
        pass


class OperatorRegressionYZ(MeanRegressionYZ):
    def __init__(self):
        super().__init__()
        self.conditional_mean_operator = ConditionalMeanOperator()
    
    def fit(
        self,
        Z: np.ndarray,
        Z_loop: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        assert Y.ndim == 1
        Y = Y.reshape(-1, 1)
        best_weight, best_loss = \
                self.conditional_mean_operator.find_best_regularization_weight(Z, Y)
        logger.debug(
            f"Best conditional mean Y|Z loss: {best_loss}, with weight "
            + f"{best_weight}"
        )
        self.conditional_mean_operator.loop_fit(Z, Z_loop)
        logger.info("Conditional Mean Regressor of Y|Z fitted.")

    def predict(
        self,
        z: np.ndarray,
        Y: np.ndarray,
        i: int,
        for_loop: bool = False,
    ) -> np.ndarray:
        if for_loop:
            return self.conditional_mean_operator.loop_predict(Y, i)
        else:
            return self.conditional_mean_operator.predict(Y, z)


class LogisticRegressionYZ(MeanRegressionYZ):
    def __init__(self):
        pass
