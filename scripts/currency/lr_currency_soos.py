import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.models.lr import LRegression

from src.utils import series_to_supervised

# TODO: Add description! Mention datasources

# =====================================================================================================================
# GET DATA
# =====================================================================================================================

# Set number of features
n_features = 3

# Get data path or create a directory if it does not exist
# TODO: This is hacky. Need to fix
pathlib.Path(os.path.join(os.path.dirname(os.getcwd()), "..", "data")).mkdir(parents=True, exist_ok=True)
data_path = os.path.join(os.path.dirname(os.getcwd()), "..", "data")

# Check if file exists
if not os.path.exists(os.path.join(data_path, "usd_zar.csv")):
    raise ValueError("No data in data folder!")

# Get currency data
currency_data = pd.read_csv(os.path.join(data_path, "usd_zar.csv"), index_col=0)
currency_data = currency_data.to_frame().ffill().dropna()

# Make data stationary
currency_data = currency_data.pct_change()

# Create supervised learning problem
currency_data = currency_data.fillna(0.0)

# Create training and testing data
x_train, x_test, y_train, y_test = train_test_split(currency_data.iloc[:, :-1], currency_data.iloc[:, -1],
                                                    test_size=0.1, random_state=1, shuffle=False)

# Create validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=1, shuffle=False)


# =====================================================================================================================
# BUILD MODEL
# =====================================================================================================================


# Create LR
lr = LRegression(name="lr_currency_nwf")

# Train LR model from scratch
lr.train(x_train, y_train, x_val, y_val, load_model=False)

# Predict
predicted = lr.predict(x_test)

# Create dataframe for predicted values
pred_df = pd.DataFrame(np.column_stack([np.squeeze(predicted), y_test]))
pred_df.columns = ["PRED", "TRUE"]

# Plot predicted values
pred_df.plot()
plt.show()
plt.close()

lr_predictions_cumulative = pred_df.fillna(0.0)
lr_predictions_cumulative = (1.0 + lr_predictions_cumulative).cumprod()
lr_predictions_cumulative = lr_predictions_cumulative.apply(lambda x: np.log(x), axis=0)
lr_predictions_cumulative.plot()
plt.title("Cumulative Return: Linear Regression")
plt.show()
plt.close()
