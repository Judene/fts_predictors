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
# GARCH
# =============================================================================
ds_Garch = newData.copy()

#DataIndex
ds_Garch['Date'] = pd.to_datetime(ds_Garch['Date'])
ds_Garch = ds_Garch.set_index('Date')
ds_Garch.head()

from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARIMA

yGarch = ds_Garch.drop(['Open', 'High','Low','Price','Vol.'], axis = 1)

#Look at acf and pacf plots (just for interest sake)
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

acf = plot_acf(yGarch['Change'], lags=30)
pacf = plot_pacf(yGarch['Change'], lags=30)
acf.set_figheight(5)
acf.set_figwidth(15)
pacf.set_figheight(5)
pacf.set_figwidth(15)
plt.show()

from scipy.stats import probplot

def ts_plot(residuals, stan_residuals, lags=50):
    residuals.plot(title='GARCH Residuals', figsize=(15, 10))
    plt.show()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax[0].set_title('GARCH Standardized Residuals KDE')
    ax[1].set_title('GARCH Standardized Resduals Probability Plot')    
    residuals.plot(kind='kde', ax=ax[0])
    probplot(stan_residuals, dist='norm', plot=ax[1])
    plt.show()
    acf = plot_acf(stan_residuals, lags=lags)
    pacf = plot_pacf(stan_residuals, lags=lags)
    acf.suptitle('GARCH Model Standardized Residual Autocorrelation', fontsize=20)
    acf.set_figheight(5)
    acf.set_figwidth(15)
    pacf.set_figheight(5)
    pacf.set_figwidth(15)
    plt.show()

#split data into train and training set
#split data into train and training set
train_data = yGarch[:1590]
test_data = yGarch[1590:]
#Figure
pricePrediction(train_data,test_data)


from arch import arch_model

garch = arch_model(train_data, vol='GARCH', p=1, q=1, dist='normal')
fgarch = garch.fit(disp='off') 
resid = fgarch.resid
st_resid = np.divide(resid, fgarch.conditional_volatility)
ts_plot(resid, st_resid)
fgarch.summary()

# forecast the test set
yhat = fgarch.forecast(horizon=len(test_data))



