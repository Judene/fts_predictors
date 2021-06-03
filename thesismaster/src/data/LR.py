import pandas as pd
import numpy as np
import matplotlib.pylab as plt
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')
import statsmodels
import statsmodels.api as sm
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import*
import datetime as ddt

from generalFunc import exploration, stationaryExploration, ADF_test, pricePrediction, reportPerformance

# =============================================================================
# #Import files
# =============================================================================
rawData = pd.read_csv(r'C:\Users\ELNA SIMONIS\Documents\MEng\2021\Data\Gold2015.csv')

# =============================================================================
# Exploration of data
#Count of missing values, change of data index, rename single column, 
#descriptive statistics, ascending order 
# =============================================================================
newData = exploration(rawData)
# =============================================================================
# Check for stationarity
#Figure of time series
#Seasonal decomposition
#Stat test by looking at mean and variance
# =============================================================================
stationaryExploration(newData)
# =============================================================================
# Check for stationarity with the dickey fuller test
# =============================================================================
change =  newData['GoldPrice']
#Dickey Fuller test
result = ADF_test(change,'raw data')

if result == 0:
    newData['Difference'] = newData['Price'].diff()
    stationaryExploration(newData)   
    ADF_test(newData['Difference'],'raw data')
else:
    print("The time series does not require further differencing")

#Final Cleaning of dataset
newData = newData.dropna()
newData.head()

# =============================================================================
# Modelling Linear Regression
# =============================================================================
LR = newData.copy()

#DataIndex
LR['Date'] = pd.to_datetime(LR['Date'])
LR = rawData.set_index('Date')
LR = LR.iloc[::-1]
#LR.head()
#Date transformation
#LR['Date']=pd.to_datetime(LR['Date'])
#LR['Date']=LR['Date'].map(ddt.datetime.toordinal)

import sklearn.preprocessing


def normalize_data(exchange):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    exchange['Price'] = min_max_scaler.fit_transform(exchange.Price.values.reshape(-1,1))
    exchange['Open'] = min_max_scaler.fit_transform(exchange.Open.values.reshape(-1,1))
    exchange['High'] = min_max_scaler.fit_transform(exchange.High.values.reshape(-1,1))
    exchange['Low'] = min_max_scaler.fit_transform(exchange.Low.values.reshape(-1,1))
    return exchange
LR = normalize_data(LR)


X = LR.drop(['Price','Change','Vol.'], axis = 1)
Y = LR.drop(['Price', 'Open', 'High', 'Low', 'Vol.'], axis = 1)

train_X = X[:1500]
train_Y = Y[:1500]
test_X = X[1500:]
test_Y = Y[1500:]

reg=LinearRegression()     #initiating linearregression
reg.fit(train_X,train_Y)

Intercept=reg.intercept_
Coefficients=reg.coef_

# Forecast
fc= reg.predict(test_X)

# Make as pandas series
#fc_series = pd.Series(fc, index=test_Y.index)

# Plot
pricePrediction(test_Y,fc)
#Performance Measure
reportPerformance(test_Y,fc)


