import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


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
data_path = "C:/Users/ELNA SIMONIS/Documents/MEng/2021/Data/data"

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

#optimal parameters
svr = SVR()
parameters = {
    "kernel": ["linear","rbf"],
    "C":[0.001, 0.01, 0.1, 1, 10]
}
cv = GridSearchCV(svr,parameters,cv=10)
cv.fit(x_train,y_train) 
cv.best_params_

cv.best_params_[1]

#BuildModel
#Optimal parameters are found from the section above and inserted here
regressor = SVR(**cv.best_params_)
regressor.fit(x_train,y_train)#Create and train an SVR model


#Predict
y_pred = regressor.predict(x_val)

# Create dataframe for predicted values
pred_df = pd.DataFrame(np.column_stack([np.squeeze(y_pred), y_val]))
pred_df.columns = ["PRED", "TRUE"]

# Plot predicted values
pred_df.plot()
plt.show()
plt.close()


