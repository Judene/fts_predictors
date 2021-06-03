import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math


class WalkForwardPredictor:

    def __init__(self, model, start_date="2015-05-01", end_date="2021-04-01", input_pct_change=1,
                 output_pct_change=1, window_size=252, frequency=7, prediction_length=21,
                 validation_size=10, sliding_window=True, random_validation=False):
        """
        This is a simple walk-forward model trainer. Given a model (sklearn models for now), it will iteratively train
        and predict the model through time and accumulate the out-of-sample predictions. From this, it will also
        calculate the error and validation score through time.
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

        # We need to make out data stationary
        #input_data = input_data.pct_change(periods=self.input_pct_change)
        #output_data = output_data.pct_change(periods=self.output_pct_change)

        # Fix INFs
        input_data[np.isinf(input_data)] = 0.0
        input_data[np.isinf(input_data)] = 0.0

        # Initialise our result DataFrames
        predictions = pd.DataFrame(index=input_data.index, columns=["predictions"])
        val_score = pd.DataFrame(index=input_data.index, columns=["val_score"])
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
                t_model = self.model.fit(x_train, y_train)
                pred = t_model.predict(x_test)

                if len(pred) == 1:
                    predictions.loc[curr_t, "predictions"] = np.squeeze(pred)
                else:
                    predictions.loc[curr_t, "predictions"] = np.squeeze(pred)[-1]

                val_score.loc[curr_t, "val_score"] = t_model.score(x_validation, y_validation)

            else:
                pred = t_model.predict(x_test)
                if len(pred) == 1:
                    predictions.loc[curr_t, "predictions"] = np.squeeze(pred)
                else:
                    predictions.loc[curr_t, "predictions"] = np.squeeze(pred)[-1]

                val_score.loc[curr_t, "val_score"] = t_model.score(x_validation, y_validation)

            # Adjust for sliding window
            if self.sliding_window:
                step_size += 1

        error = pd.DataFrame(
            np.squeeze(predictions.shift(self.prediction_length).values) - np.squeeze(output_data.values),
            index=predictions.index, columns=["error"])
        
            
        
        return predictions, val_score, error

# ======================================================================================================================
from generalFunc import exploration, stationaryExploration, ADF_test, pricePrediction, reportPerformance, cleaning
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

    # Get data (USDZAR)
    # Note: ignore the location I used. This is simply where I stored the data. Read your copy of the historical USDZAR
    # in. Adding index_col=0 ensure that we obtain the dates instead of integer indices.
    original = pd.read_csv(r'C:\Users\ELNA SIMONIS\Documents\MEng\2021\Data\Gold2015.csv')
    zar_currency = pd.read_csv(r'C:\Users\ELNA SIMONIS\Documents\MEng\2021\Data\Gold2015.csv',
        index_col=0)
    
    #Formatting of attributes
    # Rename single column
    zar_currency.rename(columns = {"Change %":"Change"}, inplace="True")
    zar_currency.rename(columns = {"Vol.":"Volume"}, inplace="True")
    zar_currency.head(1)    
    #%remove % sign
    zar_currency['Change'] = zar_currency['Change'].str.replace('%','').astype(np.float64)
    #ds_SVR['Volume'] = ds_SVR['Volume'].str.replace('K','').astype(np.float64)


    input_data = zar_currency.drop(['Price','Change','Volume'], axis = 1)
    zar_currency = zar_currency.drop(['Price', 'Open', 'High', 'Low', 'Volume'], axis = 1)
    
    #input_data = cleaning(input_data)
    zar_currency = cleaning(zar_currency)
    
    #Feature scaling
    std_x=StandardScaler()
    #std_y=StandardScaler()
    input_data = std_x.fit_transform(input_data)
    input_data = pd.DataFrame(input_data, 
             columns=['Open', 
                      'High',
                      'Low'])
    
    input_data['Date'] = original['Date']
    input_data = input_data[['Date', 'Open', 'High', 'Low']]
    input_data = input_data.iloc[::-1]
    input_data = input_data.set_index('Date')     

    
    # Initiate our model
    model = WalkForwardPredictor(model=SVR(kernel='linear', C=10), start_date="2015-05-01", end_date="2021-04-01",
                                 input_pct_change=1, output_pct_change=1, window_size=252, frequency=7,
                                 prediction_length=21, validation_size=10, sliding_window=True,
                                 random_validation=False)

    # Train our model through time, and obtain the predictions and errors
    tree_predictions, tree_val_score, tree_error = model.train_and_predict(input_data, zar_currency)
    print("SVR: With Side Info")
    print(np.mean(tree_val_score))
    print("MSE",np.mean(tree_error ** 2.0))
    print("RMSE",np.square(np.mean(tree_error ** 2.0)))

    # Create a plot of our errors through time
    tree_error = tree_error ** 2.0
    tree_error = tree_error.set_index(pd.to_datetime(tree_error.index))
    tree_error.plot()
    plt.title('MSE Gold % Change: Level 0')
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.show()
    plt.close()