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


# Create MLP model
arima_model = ARIMA(name="arima_gold_wf")

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

# sav_dates = pd.DataFrame(mlp_error)
# sav_dates = sav_dates.reset_index()
#
# saved = pd.read_csv(r'C:/Users/ELNA SIMONIS/Documents/Results/TESTING.csv')
# saved = saved.drop(['Unnamed: 0'], axis=1)
#
# saved['Dates'] = sav_dates['Date']
# saved = saved.set_index('Dates')
# saved['error'] = saved['TRUE'] - saved['PRED']
# saved = saved.dropna()
#
# # Calculate RMSE
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from math import sqrt
#
# mse = mean_squared_error(saved['TRUE'], saved['PRED'])
# rmse = sqrt(mean_squared_error(saved['TRUE'], saved['PRED']))
# mae = mean_absolute_error(saved['TRUE'], saved['PRED'])
#
# # Create a plot of our errors through time
#
# plt.figure(figsize=(10, 5))
# figuur = saved['error'] ** 2.0
# figuur.plot(color='blue')
# plt.xlabel('Dates', fontsize=15, fontweight='bold', color='black')
# plt.ylabel('Error', fontsize=15, fontweight='bold', color='black')
# plt.yticks(fontsize=10)
# plt.xticks(fontsize=10)
# plt.show()
# plt.close()