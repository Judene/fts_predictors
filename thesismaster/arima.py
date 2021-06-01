# =============================================================================
# #Import packages
#https://www.kaggle.com/nageshsingh/stock-market-forecasting-arima
# =============================================================================
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


import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from general_func import exploration, stationaryExploration, ADF_test, pricePrediction, reportPerformance
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
change =  newData['Price']
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
# Modelling
# =============================================================================
# =============================================================================
# ARIMA
# =============================================================================
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pmd
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima

ds_Arima = newData
#DataIndex
ds_Arima['Date'] = pd.to_datetime(ds_Arima['Date'])
ds_Arima = ds_Arima.set_index('Date')
ds_Arima.head()

yarima = ds_Arima.drop(['Open', 'High','Low','Change','Vol.'], axis = 1)

#split data into train and training set
train_data = yarima[:1400]
test_data = yarima[1400:]

#Figure
pricePrediction(train_data,test_data)

#modelling
model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
print(model_autoARIMA.summary())
model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()

ARorder = model_autoARIMA.order

#Modeling
# Build Model
model = ARIMA(train_data, order=ARorder)  
fitted = model.fit(disp=-1)  
print(fitted.summary())

# Forecast
fc, se, conf = fitted.forecast(len(test_data), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test_data.index)
# Plot
pricePrediction(test_data,fc_series)
#Performance Measure
reportPerformance(test_data,fc_series)

