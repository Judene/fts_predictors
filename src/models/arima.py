import numpy as np
import pandas as pd

from pathlib import Path
import pickle

from pmdarima.arima import auto_arima

import src.utils as utils

from src.models.fts_model import FTSModel


class ARIMA(FTSModel):
    """
    An implementation of an auto-ARIMA model
    """
    def __init__(self, name: str, **kwargs):
        """
        :param name: the user specified name given for the model
        """
        self.name = name

        # Model parameters
        self.start_p = 2
        self.d = None
        self.start_q = 2
        self.max_p = 5
        self.max_d = 2
        self.max_q = 5
        self.start_P = 1
        self.D = None
        self.start_Q = 1
        self.max_P = 2
        self.max_D = 1
        self.max_Q = 2
        self.max_order = 5
        self.m = 1
        self.seasonal = True
        self.stationary = False
        self.information_criterion = 'aic'
        self.alpha = 0.05
        self.test = 'kpss'
        self.seasonal_test = 'ocsb'
        self.stepwise = True
        self.n_jobs = 1
        self.start_params = None
        self.trend = None
        self.method = 'lbfgs'
        self.maxiter = 50
        self.offset_test_args = None
        self.seasonal_test_args = None
        self.suppress_warnings = True
        self.error_action = 'trace'
        self.trace = False
        self.random = False
        self.random_state = None
        self.n_fits = 10
        self.return_valid_fits = False
        self.out_of_sample_size = 0
        self.scoring = 'mse'
        self.scoring_args = None
        self.with_intercept = 'auto'

        self.model = None

        # Call the parent class constructor to initialize the object's variables
        super().__init__(name, None, **kwargs)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray,
              load_model=True) -> None:

        if load_model:
            self.load()

        else:
            self.model = auto_arima(y_train, x_train, start_p=self.start_p, d=self.d, start_q=self.start_q,
                                    max_p=self.max_p, max_d=self.max_d, max_q=self.max_q, start_P=self.start_P,
                                    D=self.D, start_Q=self.start_Q, max_P=self.max_P, max_D=self.max_D,
                                    max_Q=self.max_Q, max_order=self.max_order, m=self.m, seasonal=self.seasonal,
                                    stationary=self.stationary, information_criterion=self.information_criterion,
                                    alpha=self.alpha, test=self.test, seasonal_test=self.seasonal_test,
                                    stepwise=self.stepwise, n_jobs=self.n_jobs, start_params=self.start_params,
                                    trend=self.trend, method=self.method, maxiter=self.maxiter,
                                    offset_test_args=self.offset_test_args,
                                    seasonal_test_args=self.seasonal_test_args,
                                    suppress_warnings=self.suppress_warnings,
                                    error_action=self.error_action, trace=self.trace,
                                    random=self.random, random_state=self.random_state, n_fits=self.n_fits,
                                    return_valid_fits=self.return_valid_fits,
                                    out_of_sample_size=self.out_of_sample_size, scoring=self.scoring,
                                    scoring_args=self.scoring_args, with_intercept=self.with_intercept)

            self.save()

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> tuple:
        # TODO: Double-check!
        score = self.model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=0)
        test_loss = score[0]
        test_accuracy = score[1]
        print("accuracy: {:.2f}% | loss: {}".format(100 * test_accuracy, test_loss))
        print("SCORE: ", score)
        return test_loss, test_accuracy

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
        prediction = self.model.predict(n_periods=np.shape(x)[0], X=x)
        return prediction
