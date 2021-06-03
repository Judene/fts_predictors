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

from generalFunc import exploration, stationaryExploration, ADF_test, pricePrediction, reportPerformance, cleaning

# =============================================================================
# #Import files
# =============================================================================
rawData = pd.read_csv(r'C:\Users\ELNA SIMONIS\Documents\MEng\2021\Data\Gold2015.csv')

# =============================================================================
# Exploration of data
#Count of missing values, change of data index, rename single column, 
#descriptive statistics, ascending order 
# =============================================================================
newData = cleaning(rawData)
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
# SVR
# =============================================================================
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
ds_SVR = newData.copy()

#DataIndex
ds_SVR['Date'] = pd.to_datetime(ds_SVR['Date'])
ds_SVR = ds_SVR.set_index('Date')
ds_SVR.head()

#Formatting of attributes
# Rename single column
ds_SVR.rename(columns = {"Change %":"Change"}, inplace="True")
ds_SVR.rename(columns = {"Vol.":"Volume"}, inplace="True")
ds_SVR.head(1)    
#%remove % sign
ds_SVR['Change'] = ds_SVR['Change'].str.replace('%','').astype(np.float64)
#ds_SVR['Volume'] = ds_SVR['Volume'].str.replace('K','').astype(np.float64)


xSVR = ds_SVR.drop(['Price','Change','Volume'], axis = 1)
ySVR = ds_SVR.drop(['Price', 'Open', 'High', 'Low', 'Volume'], axis = 1)

#Vraag oor hoe mens scaling toepas in READ.Me file
#Feature scaling
std_x=StandardScaler()
#std_y=StandardScaler()
X = std_x.fit_transform(xSVR)
#y = std_y.fit_transform(ySVR)
ySVR = ySVR.values
y = ySVR
train_X = X[:1500]
train_Y = y[:1500]
test_X = X[1500:]
test_Y = y[1500:]

#optimal parameters
svr = SVR()
parameters = {
    "kernel": ["linear","rbf"],
    "C":[0.001, 0.01, 0.1, 1, 10]
}
cv = GridSearchCV(svr,parameters,cv=10)
cv.fit(train_X,train_Y) 
cv.best_params_

#BuildModel
#Optimal parameters are found from the section above and inserted here
regressor = SVR(kernel='linear', C=10)
regressor.fit(train_X,train_Y)#Create and train an SVR model

y_pred = regressor.predict(test_X)
#y_pred= std_y.inverse_transform(y_pred)

#Create table for predicted and actual values
# Make as pandas series
finals = pd.DataFrame(y_pred)
final = pd.DataFrame(test_Y)
finals.rename(columns = {0:"predict"}, inplace="True")
final.rename(columns = {0:"actual"}, inplace="True")
finals.head(1)    

# Plot
pricePrediction(final['actual'],finals['predict'])
#Performance Measure
reportPerformance(final['actual'],finals['predict'])
