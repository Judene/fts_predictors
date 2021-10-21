import abc

import numpy as np


class FTSModel(metaclass=abc.ABCMeta):
    """
    The `FTSModel` class is an abstract class which exposes a simple API for creating FTS (Financial Time Series)
    models.
    """
    @abc.abstractmethod
    def __init__(self, name: str, model, *args, **kwargs):
        """
        :param name: the user specified name given for the model
        :param model: the model
        """
        self.name = name
        self.model = model

    @abc.abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray,
              load_model=True) -> None:
        raise NotImplementedError("Not Implemented")

    @abc.abstractmethod
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> tuple:
        raise NotImplementedError("Not Implemented")

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError("Not Implemented")

    @abc.abstractmethod
    def load(self):
        raise NotImplementedError("Not Implemented")

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Not Implemented")
