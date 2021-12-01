import os
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

from src.models.walk_forward_predictor import WalkForwardPredictor
from src.models.arima import ARIMA

from src.utils import series_to_supervised

# Get data path or create a directory if it does not exist
# TODO: This is hacky. Need to fix
pathlib.Path(os.path.join(os.path.dirname(os.getcwd()), "..", "data")).mkdir(parents=True, exist_ok=True)
data_path = os.path.join(os.path.dirname(os.getcwd()), "..", "data")

# Check if file exists
if not os.path.exists(os.path.join(data_path, "s&p500_index.csv")):
    raise ValueError("No data in data folder!")

# Get index data
index_data = pd.read_csv(os.path.join(data_path, "s&p500_index.csv"), index_col=0)
index_data = index_data.to_frame().ffill().dropna()
dates = index_data

n_features = 1

index_data = series_to_supervised(index_data, n_in=n_features, n_out=1)
input_data = index_data.drop(['var1(t)'], axis=1)
output_data = index_data.drop(['var1(t-1)'], axis=1)


# Create ARIMA model
arima_model = ARIMA(name="arima_index_wf")

# Initiate our model
wf_model = WalkForwardPredictor(model=arima_model, start_date="2004-11-08", end_date="2021-06-01",
                                input_pct_change=1, output_pct_change=1, window_size=252, frequency=42,
                                prediction_length=10, validation_size=2, sliding_window=False,
                                random_validation=False, train_from_scratch=True)

# Train our model through time, and obtain the predictions and errors
arima_predictions, arima_error = wf_model.train_and_predict(input_data, output_data)

print("ARIMA Walk Forward")

print(arima_predictions)
print(arima_error)
