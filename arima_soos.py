#Import necessary packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
import statsmodels.api as sm
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
import pmdarima as pmd
from pmdarima.arima import auto_arima

# =====================================================================================================================
# FUNCTIONS
# =====================================================================================================================


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# =====================================================================================================================
# GET DATA
# =====================================================================================================================

# Set number of features
n_features = 2

# Get data path
data_path = "//home//zander//projects//judene_thesis//Master_Thesis//thesismaster//data"

# Get gold data
gold_etf_data = pd.read_csv(os.path.join(data_path, "local_etfs_close.csv"), index_col=0)
gold_etf_data = gold_etf_data["GLD"].to_frame().ffill().dropna()

# Make data stationary
gold_etf_data = gold_etf_data.pct_change()

# Create supervised learning problem
gold_etf_data = series_to_supervised(gold_etf_data.values, n_in=n_features, n_out=1)

# Create training and testing data
x_train, x_test, y_train, y_test = train_test_split(gold_etf_data.iloc[:, :-1], gold_etf_data.iloc[:, -1],
                                                    test_size=0.1, random_state=1, shuffle=False)

# Create validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=1, shuffle=False)

# =============================================================================
# Build model
# =============================================================================

# =============================================================================
# Build the model ARIMAX
# =============================================================================
# Multipletime series variables
exog = ["var1(t-2)", "var1(t-1)"]

# Build the model
model = auto_arima(y_train, exogenous=x_train[exog], trace=True, error_action="ignore", suppress_warnings=True)
model.fit(y_train, exogenous=x_train[exog])

# Predict
predicted = model.predict(n_periods=len(y_test), exogenous=x_test[exog])


# Create dataframe for predicted values
pred_df = pd.DataFrame(np.column_stack([np.squeeze(predicted), y_test]))
pred_df.columns = ["PRED", "TRUE"]
pred_df.to_csv('C:/Users/ELNA SIMONIS/Documents/Results/Bitcoin_ARIMA.csv')

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

#Calculate RMSE

mse = mean_squared_error(pred_df['TRUE'], predicted)
rmse = sqrt(mean_squared_error(pred_df['TRUE'], predicted))
mae = mean_absolute_error(pred_df['TRUE'], predicted)

#Other calculations
dataset = pred_df.copy()

def PredStrategy(c):
    if c['PRED'] > 0:
        return 1
    else:
        return -1
    
def TrueStrategy(c):
    if c['TRUE'] > 0:
        return 1
    else:
        return -1
    

dataset['PredStrategy'] = dataset.apply(PredStrategy, axis = 1)
dataset['TrueStrategy'] = dataset.apply(TrueStrategy, axis = 1)
dataset['Accuracy'] = dataset['PredStrategy']*dataset['TrueStrategy']


#Movement over time
balance_init = 1000
balance = []
changes = dataset['PRED'].values

move = balance_init*(100+changes[0])/100
balance.append(move)
carryOver = move

for i in range(len(changes)-1):
    move = carryOver*(100+changes[i+1])/100
    balance.append(move)
    carryOver = move

roi = (balance[-1]- balance_init)/balance_init*100

dataset['MovePred'] = balance

dataset['Profit'] = dataset['MovePred']-balance_init

averagePorfit = dataset['Profit'].mean()
stdProfit = dataset['Profit'].std()
sharpe = averagePorfit/stdProfit

#Accuracy
dataset = dataset.apply(lambda x : True
            if x['Accuracy'] == 1 else False, axis = 1)

num_rows = len(dataset[dataset == True].index)
accuracy = num_rows/len(dataset.index)


# Plot predicted values
pred_df.plot()
plt.show()
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(pred_df['TRUE'], color = 'green', label = 'Actual values')
plt.plot(pred_df['PRED'], color = 'black', label = 'Predicted values')
plt.xlabel('Time', fontsize=10, fontweight='bold', color = 'black')
plt.ylabel('Close price change (%)', fontsize=10, fontweight='bold', color = 'black')
ax = plt.axes()
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
ax.legend()
ax.set_facecolor("white")
ax.tick_params(axis="x", colors="black")
ax.tick_params(axis="y", colors="black")
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

plt.show()

print(mse, rmse, mae)
