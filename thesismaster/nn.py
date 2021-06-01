# TODO: Some notes in terms of styling
# Check spacing
# Check use of caps, comments and file namings
# Don't use caps for variable names, e.g. X_train. Rather x_train

# =============================================================================
# Import packages
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

plt.rcParams["figure.figsize"] = (20, 10)
plt.style.use("ggplot")

import statsmodels
import statsmodels.api as sm
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import lag_plot

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import sklearn
import datetime
import os
import sklearn.preprocessing
from sklearn.metrics import r2_score

from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential


# =============================================================================
# Import files
# =============================================================================

exchange = pd.read_csv(r"C:\Users\ELNA SIMONIS\Documents\MEng\2021\Data\Gold2015.csv")

# =============================================================================
# Exploration of data
# =============================================================================

# Count missing values
if exchange.isnull().sum().sum() > 0:
    print("There are missing values in this dataset")
    exchange = exchange.dropna()
else:
    print("There are no missing values in this dataset")

# Rename single column
exchange.rename(columns = {"Change %":"Change"}, inplace="True")
exchange.head(1)

# Remove % sign
exchange["Change"] = exchange["Change"].str.replace("%","").astype(np.float64)
    
# Descriptive
types = exchange.dtypes
des_stat = exchange.describe()

print(types)
print(des_stat)

exchange["Date"] = pd.to_datetime(exchange["Date"])
exchange = exchange.set_index("Date")
exchange.head()

df_ex = exchange.copy()

exchange = exchange.iloc[::-1]

# =============================================================================
# Manipulate the data
# =============================================================================

# Function for min-max normalization of stock
def normalize_data(exchange):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    exchange["Open"] = min_max_scaler.fit_transform(exchange.Open.values.reshape(-1,1))
    exchange["High"] = min_max_scaler.fit_transform(exchange.High.values.reshape(-1,1))
    exchange["Low"] = min_max_scaler.fit_transform(exchange.Low.values.reshape(-1,1))
    exchange["Price"] = min_max_scaler.fit_transform(exchange.Price.values.reshape(-1,1))
    #exchange["Change"] = min_max_scaler.fit_transform(exchange.Change.values.reshape(-1,1))
    return exchange

exchange_norm = normalize_data(exchange)
exchange_norm.shape

# Choose sequence length
seq_len = 20

stock = exchange_norm


def load_data(stock, seq_len):
    stock_x = stock.drop(["Change", "Vol.", "Price"], axis=1)
    stock_y = stock.drop(["Open", "High", "Low", "Price", "Vol."], axis=1)

    x_train = []
    y_train = []
    for i in range(seq_len, len(stock)):
        x_train.append(stock_x.iloc[i - seq_len: i, 0:3])
        y_train.append(stock_y.iloc[i, 0:1])

    # 1 last 6189 days are going to be used in test
    X_test = x_train[1550:]
    y_test = y_train[1550:]

    # 2 first 110000 days are going to be used in training
    x_train = x_train[:1550]
    y_train = y_train[:1550]

    # 3 convert to numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 4 reshape data to input into RNN models
    x_train = np.reshape(x_train, (1550, seq_len, 3))

    X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 3))

    return [x_train, y_train, X_test, y_test]

x_train, y_train, X_test, y_test = load_data(stock, seq_len)
    
print("x_train.shape = ",x_train.shape)
print("y_train.shape = ", y_train.shape)
print("X_test.shape = ", X_test.shape)
print("y_test.shape = ",y_test.shape)


# =============================================================================
# Simple RNN model
# =============================================================================

rnn_model = Sequential()

rnn_model.add(SimpleRNN(50,activation="relu",return_sequences=True, input_shape=(x_train.shape[1],3)))
rnn_model.add(Dropout(0.15))

rnn_model.add(SimpleRNN(50,activation="relu",return_sequences=True))
rnn_model.add(Dropout(0.15))

rnn_model.add(SimpleRNN(50,activation="relu",return_sequences=False))
rnn_model.add(Dropout(0.15))

rnn_model.add(Dense(1))

rnn_model.summary()


# =============================================================================
# Run the model
# =============================================================================
rnn_model.compile(optimizer="adam",loss="MSE")
rnn_model.fit(x_train, y_train, epochs=10, batch_size=20)

rnn_predictions = rnn_model.predict(X_test)

rnn_score = r2_score(y_test,rnn_predictions)
print("R2 Score of RNN model = ",rnn_score)

MSE = mean_squared_error(y_true = y_test, y_pred = rnn_predictions)
print("MSE Score of RNN model = ",MSE)

# =============================================================================
# Plot the model
# =============================================================================
from sklearn.preprocessing import MinMaxScaler

std_y = MinMaxScaler(feature_range=(-1,1))
finalPred= std_y.inverse_transform(rnn_predictions)
aactualy = std_y.inverse_transform(y_test)

from general_func import exploration, stationaryExploration, ADF_test, pricePrediction, reportPerformance

pricePrediction(y_test,rnn_predictions)

# =============================================================================
# LSTM
# =============================================================================
regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 3)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.25))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.25))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.25))

regressor.add(Dense(units = 1))

regressor.summary()

regressor.compile(optimizer = "adam", loss = "mean_squared_error")

regressor.fit(x_train, y_train, epochs=10, batch_size=20)
lstm_predictions = regressor.predict(X_test)

rnn_score = r2_score(y_test,lstm_predictions)
print("R2 Score of RNN model = ",rnn_score)

MSE = mean_squared_error(y_true = y_test, y_pred = lstm_predictions)
print("MSE Score of RNN model = ",MSE)

# =============================================================================
# Plot the model
# =============================================================================
from sklearn.preprocessing import MinMaxScaler

std_y = MinMaxScaler(feature_range=(-1,1))
finalPred= std_y.inverse_transform(rnn_predictions)
aactualy = std_y.inverse_transform(y_test)

from general_func import exploration, stationaryExploration, ADF_test, pricePrediction, reportPerformance

pricePrediction(y_test,lstm_predictions)

