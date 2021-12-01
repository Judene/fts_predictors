import numpy as np
import pandas as pd

from pathlib import Path
import pickle

from sklearn.svm import SVR

import src.utils as utils

from src.models.fts_model import FTSModel


class SupportVectorRegression(FTSModel):
    """
    An implementation of a Support Vector Regression model
    """
    def __init__(self, name: str, **kwargs):
        """
        :param name: the user specified name given for the model
        """
        self.name = name

        # Model parameters
        # See: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        self.kernel = 'rbf'
        self.degree = 3
        self.gamma = 'scale'
        self.coef0 = 0.0
        self.tol = 0.001
        self.C = 1.0
        self.epsilon = 0.1
        self.shrinking = True
        self.cache_size = 200
        self.verbose = False
        self.max_iter = -1

        self.__dict__.update(kwargs)

        svr = SVR(kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0,
                  tol=self.tol, C=self.C, epsilon=self.epsilon, shrinking=self.shrinking,
                  cache_size=self.cache_size, verbose=self.verbose, max_iter=self.max_iter)

        self.model = svr

        # Call the parent class constructor to initialize the object's variables
        super().__init__(name, svr, **kwargs)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray,
              load_model=True) -> None:

        if load_model:
            self.load()

        else:
            self.model.fit(x_train, y_train)
            self.save()

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        # TODO: Double-check!
        score = self.model.score(x_test, y_test)
        print("SCORE: ", score)
        return score

    def save(self):
        project_dir = Path(__file__).resolve().parents[2]
        models_dir = str(project_dir) + '/models/' + self.name + '/'
        utils.check_folder(models_dir)
        with open(models_dir + self.name + ".pkl", "wb") as pkl:
            pickle.dump(self.model, pkl)
        print("Saved model to disk")

    def load(self):
        try:
            project_dir = Path(__file__).resolve().parents[2]
            models_dir = str(project_dir) + '/models/' + self.name + '/'
            utils.check_folder(models_dir)
            with open(models_dir + self.name + ".pkl", "wb") as pkl:
                self.model = pickle.load(pkl)
            print("Loaded " + self.name + " model from disk")

        except ValueError as e:
            print("No saved model found. Check file name or train from scratch")

    def predict(self, x: (np.ndarray, pd.DataFrame)) -> np.ndarray:
        prediction = self.model.predict(x)
        return prediction
