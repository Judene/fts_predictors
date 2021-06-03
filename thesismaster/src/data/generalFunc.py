# =============================================================================
# #Import packages
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

# =============================================================================
# #Import files
# =============================================================================

def exploration(rawData):
        
        #Count of missing values
    if rawData.isnull().sum().sum() > 0:
        print('There are missing values in this dataset')
        rawData = rawData.dropna()
    else:
        print('There are no missing values in this dataset')
    
    # Rename single column
    rawData.rename(columns = {"Change %":"Change"}, inplace="True")
    #rawData.rename(columns = {"Vol.":"Volume"}, inplace="True")
    rawData.head(1)    
    #%remove % sign
    rawData['Change'] = rawData['Change'].str.replace('%','').astype(np.float64)
    #rawData['Vol.'] = rawData['Vol.'].str.replace('K','').astype(np.float64)
        
    #Descriptive
    types = rawData.dtypes
    des_stat = rawData.describe()
    print(types)
    print(des_stat)
      
    #Sort ascending to descending
    rawData = rawData.iloc[::-1]
    
    #drop columns
    #exchange = exchange.drop(['Close'], axis = 1)
    
    return rawData

def cleaning(rawData):
    
    #Descriptive
    types = rawData.dtypes
    des_stat = rawData.describe()
    print(types)
    print(des_stat)
      
    #Sort ascending to descending
    rawData = rawData.iloc[::-1]
    return rawData



# =============================================================================
# Check for stationarity
# =============================================================================

def stationaryExploration(rawData):
    change =  rawData['Price']
    #Through plots
    #Normal plot
    plt.figure(figsize=(10, 7))
    plt.plot(change)
    plt.title('Financial Market', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Close', fontsize=12)
    plt.show() 
    
    #Seasonal Decomposition
    result = sm.tsa.seasonal_decompose(change, model='multiplicative', freq = 30)
    fig = plt.figure()  
    fig = result.plot()  
    fig.set_size_inches(16, 9)

    #Summary statistics
    one, two, three = np.split(
            change.sample(
            frac=1), [int(.25*len(change)),
            int(.75*len(change))])

    mean1, mean2, mean3 = one.mean(), two.mean(), three.mean()
    var1, var2, var3 = one.var(), two.var(), three.var()
    print(mean1, mean2, mean3)
    print(var1, var2, var3)
    
    return 

def ADF_test(timeseries, dataDesc):
    print(' > Is the {} stationary ?'.format(dataDesc))
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    print('Test statistic = {:.3f}'.format(dftest[0]))
    print('P-value = {:.3f}'.format(dftest[1]))
    print('Critical values :')
    confidence = []
    for k, v in dftest[4].items():
        print('\t{}: {} - The data is {} stationary with {}% confidence'.format(k, v, 'not' if v<dftest[0] else '', 100-int(k[:-1])))
        if v<dftest[0]:
            confidence = 0
        else:
            confidence = 1
    pd.plotting.lag_plot(timeseries)
    return confidence

# =============================================================================
# Final predictions and results
# =============================================================================
# Plot
def pricePrediction(test_Y,fc_series):
    
    plt.figure(figsize=(10,5), dpi=100)
    plt.plot(test_Y, color = 'blue', label='Actual')
    plt.plot(fc_series, color = 'orange',label='Predicted')
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    return

def reportPerformance(test_Y,fc_series):

    mse = mean_squared_error(test_Y, fc_series)
    print('MSE: '+str(mse))
    mae = mean_absolute_error(test_Y, fc_series)
    print('MAE: '+str(mae))
    rmse = math.sqrt(mean_squared_error(test_Y, fc_series))
    print('RMSE: '+str(rmse))
    return mse, mae, rmse


