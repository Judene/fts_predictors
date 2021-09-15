import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
import pmdarima as pmd
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

class WalkForwardMLPPredictor:

    def __init__(self, model, start_date="2000-01-03", end_date="2020-10-13", input_pct_change=1,
                 output_pct_change=1, window_size=252, frequency=7, prediction_length=20,
                 validation_size=10, sliding_window=True, random_validation=False):
        """
        TODO: Update description
        :param model: A sklearn model instance
        :param start_date: start date of the simulation
        :param end_date: end date of the simulation
        :param input_pct_change: how many periods to use for calculation of the input returns
        :param output_pct_change: how many periods to use for calculation of the output returns
        :param window_size: number of days in the window
        :param frequency: how often to retrain the model
        :param prediction_length: how far into the future to predict
        :param validation_size: how many datapoints in the validation set
        :param sliding_window: whether to grow or slide the window
        :param random_validation: whether to sample a random validation set
        """

        # Date parameters
        self.start_date = start_date
        self.end_date = end_date

        # Period parameter for calculating returns (stationarity)
        self.input_pct_change = input_pct_change
        self.output_pct_change = output_pct_change

        # Window Size
        self.window_size = window_size

        # Frequency of training and predicting
        self.frequency = frequency

        # Prediction length
        self.prediction_length = prediction_length

        # Define model
        self.model = model

        # Validation data to use
        self.validation_size = validation_size

        # Sliding or growing window
        self.sliding_window = sliding_window

        # To use random validation data
        self.random_validation = random_validation

    def train_and_predict(self, input_data, output_data):

        # Convert to Series to DataFrame
        if isinstance(input_data, pd.Series):
            input_data = input_data.to_frame()

        if isinstance(output_data, pd.Series):
            output_data = output_data.to_frame()

        # Date correction
        input_data = input_data.set_index(pd.to_datetime(input_data.index))
        input_data = input_data.set_index(input_data.index.strftime("%Y-%m-%d"))
        output_data = output_data.set_index(pd.to_datetime(output_data.index))
        output_data = output_data.set_index(output_data.index.strftime("%Y-%m-%d"))

        # Error checking
        if np.shape(output_data)[1] != 1:
            raise ValueError("Output data must be univariate. Multi-output prediction not supported.")
        if len(input_data) != len(output_data):
            raise ValueError("Input data and Output data should have the same length.")
        if not input_data.index.equals(output_data.index):
            raise ValueError("Please ensure that input and output have the same index. \
                              Use pandas' reindex() function if not.")
        if self.window_size >= len(input_data):
            raise ValueError("Window size cannot be larger than the size of the input data.")
        if self.validation_size >= int(self.window_size / 4.0):
            raise ValueError("Validation sample is too big.")

        # Cap our data to our specified dates
        input_data = input_data[self.start_date:self.end_date]
        output_data = output_data[self.start_date:self.end_date]

        # Fix INFs
        input_data[np.isinf(input_data)] = 0.0
        input_data[np.isinf(input_data)] = 0.0

        # Initialise our result DataFrames
        predictions = pd.DataFrame(index=input_data.index, columns=["predictions"])
        #val_score = pd.DataFrame(index=input_data.index, columns=["val_score"])
        step_size = 0
        t_model = None

        # Start main loop
        for t in range(self.window_size, len(input_data)):

            if t + self.prediction_length > len(input_data):
                break

            # Get current point in time
            curr_t = input_data.iloc[min(t + self.prediction_length - 1, len(input_data))].name

            # 1) Get WINDOW_SIZE + PREDICTION_LENGTH worth of input data
            temp_input_data = input_data[step_size:(t + self.prediction_length)]

            # 2) Shift output data forward by PREDICTION_LENGTH, and
            # get WINDOW_SIZE + PREDICTION_LENGTH worth of input data
            temp_output_data = output_data[step_size:(t + self.prediction_length)]
            temp_output_data = temp_output_data.shift(-self.prediction_length)

            # 3) Save the last S inputs for out-of-sample prediction
            x_test = temp_input_data[temp_output_data.last_valid_index():]
            x_test = x_test[1:]

            # 4) Assign the rest for training
            x_train = temp_input_data.drop(x_test.index)

            # 5) Extract a validation set
            if self.random_validation:
                x_validation = x_train.sample(self.validation_size).fillna(0.0)

            else:
                x_validation = x_train.tail(self.validation_size).fillna(0.0)

            x_train = x_train.drop(x_validation.index).fillna(0.0)
            y_validation = temp_output_data.loc[x_validation.index].fillna(0.0)
            y_train = temp_output_data.loc[x_train.index].fillna(0.0)

            # 6) Train model and/or predict
            if t % self.frequency == 0 or self.model is None:

                t_model = self.model
                try:
                    t_model.load()
                    t_model.train(x_train, y_train, x_validation, y_validation, load_model=False)
                except OSError:
                    t_model.train(x_train, y_train, x_validation, y_validation, load_model=False)

                pred = t_model.predict(x_test)
                #pred = scaler2.inverse_transform(pred)
                #pred = np.array(pred).ravel()

                if len(pred) == 1:
                    predictions.loc[curr_t, "predictions"] = np.squeeze(pred)
                else:
                    predictions.loc[curr_t, "predictions"] = np.squeeze(pred)[-1]

                # val_score.loc[curr_t, "val_score"] = t_model.score(x_validation, y_validation)

            else:
                pred = t_model.predict(x_test)
                #pred = scaler2.inverse_transform(pred)
                #pred = np.array(pred).ravel()
                
                if len(pred) == 1:
                    predictions.loc[curr_t, "predictions"] = np.squeeze(pred)
                else:
                    predictions.loc[curr_t, "predictions"] = np.squeeze(pred)[-1]

                # val_score.loc[curr_t, "val_score"] = t_model.score(x_validation, y_validation)

            # Adjust for sliding window
            if self.sliding_window:
                step_size += 1
                
        
        error = pd.DataFrame(
            np.squeeze(predictions.shift(self.prediction_length).values) - np.squeeze(output_data.values),
            index=predictions.index, columns=["error"])
        
# =============================================================================
#         output_data = scaler2.inverse_transform(output_data)
#         output_data = pd.DataFrame(output_data, columns=['TRUE'])
#         predictions = scaler2.inverse_transform(predictions)
#         predictions = pd.DataFrame(predictions, columns=['PRED'])
# =============================================================================
        
        
# =============================================================================
#         # Create dataframe for predicted values
#         pred_df = pd.DataFrame(np.column_stack([np.squeeze(predictions), np.squeeze(output_data)]))
#         pred_df.columns = ["PRED", "TRUE"]
#         pred_df.to_csv('C:/Users/ELNA SIMONIS/Documents/Results/TESTING.csv')
# =============================================================================

        return predictions, error

# ======================================================================================================================

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


if __name__ == "__main__":
    
    
    # Get data path
    data_path = "//home//zander//projects//judene_thesis//Master_Thesis//thesismaster//data"

    # Get gold data
    gold_etf_data = pd.read_csv(os.path.join(data_path, "local_etfs_close.csv"), index_col=0)
    gold_etf_data = gold_etf_data["GLD"].to_frame().ffill().dropna()
    
    gold_etf_data = gold_etf_data.pct_change()
    
    # Set number of features
    n_features = 2
    # Create supervised learning problem
    gold_etf_data = series_to_supervised(gold_etf_data.values, n_in=n_features, n_out=1)
    gold_etf_data['Date'] = datum[1:]
    gold_etf_data = gold_etf_data.set_index('Date')
    
    gold_etf_data.dtypes
    
    # Multipletime series variables
    #exog = ["var1(t-2)", "var1(t-1)"]
    
    input_data = gold_etf_data.drop(['var1(t)'], axis = 1)
    output_data = gold_etf_data.drop(['var1(t-2)', 'var1(t-1)'], axis = 1)
    
    # Build the model
    import statsmodels.api as sm
    model3=sm.tsa.ARIMA(endog=output_data,exog=input_data,order=[2,0,4])
    arima_model = auto_arima(output_data, exogenous=input_data, trace=True, error_action="ignore", suppress_warnings=True)
# =============================================================================
#     input_data = np.array(input_data)
#     output_data = np.array(output_data)
#     
#     input_data = input_data.reshape(-1,1)
#     output_data = output_data.reshape(-1,1)
# =============================================================================
    
    #model.fit(y_train, exogenous=x_train[exog])
        
    # Initiate our model
    model = WalkForwardPredictor(model=arima_model, start_date="2001-01-08", end_date="2021-07-30",
                                 input_pct_change=1, output_pct_change=1, window_size=252, frequency=7,
                                 prediction_length=21, validation_size=20, sliding_window=True,
                                 random_validation=False)

    # Train our model through time, and obtain the predictions and errors
    tree_predictions, tree_val_score, tree_error = model.train_and_predict(input_data, output_data)
    print("LR: With Side Info")
    print(np.mean(tree_val_score))
    print("MSE" , np.mean(tree_error ** 2.0))
    print("RMSE" , math.sqrt(np.mean(tree_error ** 2.0)))
    print("MAE" ,np.mean(abs(tree_error)))
    
    # Create a plot of our errors through time
    tree_error = tree_error ** 2.0
    tree_error = tree_error.set_index(pd.to_datetime(tree_error.index))
    
    plt.figure(figsize=(7, 4))
    plt.plot(tree_error, color = 'orange')
    plt.xlabel('Date', fontsize=10, fontweight='bold', color = 'black')
    plt.ylabel('Error', fontsize=10, fontweight='bold', color = 'black')
    ax = plt.axes()
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    ax.set_facecolor("white")
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()
