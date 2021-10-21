import os
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.models.walk_forward_predictor import WalkForwardPredictor
from src.models.svr import SupportVectorRegression

from src.utils import series_to_supervised

# Get data path or create a directory if it does not exist
# TODO: This is hacky. Need to fix
pathlib.Path(os.path.join(os.path.dirname(os.getcwd()), "..", "data")).mkdir(parents=True, exist_ok=True)
data_path = os.path.join(os.path.dirname(os.getcwd()), "..", "data")

# Check if file exists
if not os.path.exists(os.path.join(data_path, "local_etfs_close.csv")):
    raise ValueError("No data in data folder!")

# Get gold data
gold_etf_data = pd.read_csv(os.path.join(data_path, "local_etfs_close.csv"), index_col=0)
gold_etf_data = gold_etf_data["GLD"].to_frame().ffill().dropna()
dates = gold_etf_data

n_features = 1

gold_etf_data = series_to_supervised(gold_etf_data, n_in=n_features, n_out=1)
input_data = gold_etf_data.drop(['var1(t)'], axis=1)
output_data = gold_etf_data.drop(['var1(t-1)'], axis=1)


# Create SVR model
svr_model = SupportVectorRegression(name="svr_gold_wf")

# Initiate our model
wf_model = WalkForwardPredictor(model=svr_model, start_date="2004-11-08", end_date="2021-06-01",
                                input_pct_change=1, output_pct_change=1, window_size=252, frequency=1,
                                prediction_length=10, validation_size=2, sliding_window=True,
                                random_validation=False, train_from_scratch=True)

# Train our model through time, and obtain the predictions and errors
svr_predictions, svr_error = wf_model.train_and_predict(input_data, output_data)

print("SVR Walk Forward")

print(svr_predictions)
print(svr_error)

wf_se = svr_error ** 2.0
wf_se.plot()
plt.title("SE: Support Vector Regression")
plt.show()

svr_predictions_cumulative = svr_predictions.fillna(0.0)
svr_predictions_cumulative = (1.0 + svr_predictions_cumulative).cumprod()
svr_predictions_cumulative = svr_predictions_cumulative.apply(lambda x: np.log(x), axis=0)
svr_predictions_cumulative.plot()
plt.title("Cumulative Return: Support Vector Regression")
plt.show()
plt.close()
