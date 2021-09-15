import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

from mlp_test_code import MultiLayerPerceptron


class WalkForwardMLPPredictor:

    def __init__(self, model, start_date="2000-01-03", end_date="2020-10-13", input_pct_change=1,
                 output_pct_change=1, window_size=252, frequency=7, prediction_length=10,
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

        # We need to make out data stationary
        input_data = input_data.pct_change(periods=self.input_pct_change)
        output_data = output_data.pct_change(periods=self.output_pct_change)

        # Fix INFs
        input_data[np.isinf(input_data)] = 0.0
        input_data[np.isinf(input_data)] = 0.0

        # Initialise our result DataFrames
        predictions = pd.DataFrame(index=input_data.index, columns=["predictions"])
        # val_score = pd.DataFrame(index=input_data.index, columns=["val_score"])
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

                if len(pred) == 1:
                    predictions.loc[curr_t, "predictions"] = np.squeeze(pred)
                else:
                    predictions.loc[curr_t, "predictions"] = np.squeeze(pred)[-1]

                # val_score.loc[curr_t, "val_score"] = t_model.score(x_validation, y_validation)

            else:
                pred = t_model.predict(x_test)
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

        # Create dataframe for predicted values
        pred_df = pd.DataFrame(np.column_stack([np.squeeze(predictions), np.squeeze(output_data)]))
        pred_df.columns = ["PRED", "TRUE"]
        pred_df.to_csv('C:/Users/ELNA SIMONIS/Documents/Results/TESTING.csv')

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
    # Get gold data
    gold_etf_data = pd.read_csv(r'C:\Users\ELNA SIMONIS\Documents\MEng\2021\Data\EditedData\Bond.csv')
    gold_etf_data = gold_etf_data.ffill().dropna()

    gold_etf_data.dtypes
    gold_etf_data["Price"] = pd.to_numeric(gold_etf_data["Price"])

    datum = gold_etf_data["Date"]
    gold_etf_data = gold_etf_data.set_index('Date')

    n_features = 2
    # gold_etf_data = gold_etf_data.pct_change()
    gold_etf_data = series_to_supervised(gold_etf_data.values, n_in=n_features, n_out=1)
    input_data = gold_etf_data.drop(['var1(t)'], axis=1)
    output_data = gold_etf_data.drop(['var1(t-2)', 'var1(t-1)'], axis=1)

    output_data['Date'] = datum[1:]
    output_data = output_data.set_index('Date')
    input_data['Date'] = datum[1:]
    input_data = input_data.set_index('Date')

    input_data.dtypes
    input_data.shape

    mlp_model = MultiLayerPerceptron(
        name="mlp_gold_wf",
        num_inputs=n_features,
        num_outputs=1,
        # If true, training info is outputted to stdout
        keras_verbose=False,
        # A summary of the NN is printed to stdout
        print_model_summary=True,
        # ff_layers = [units, activation, regularization, dropout, use_bias]
        ff_layers=[
            [512, "relu", 0.0, 0.2, True, "gaussian"],
            [512, "relu", 0.0, 0.2, True, "gaussian"],
            [512, "relu", 0.0, 0.2, True, "gaussian"]
        ],
        # The final output layer's activation function
        final_activation="tanh",
        # The objective function for the NN
        objective="mse",
        # The maximum number of epochs to run
        epochs=5,
        # The batch size to use in the NN
        batch_size=64,
        # The learning rate used in optimization
        learning_rate=0.001,
        # If this many stagnant epochs are seen, stop training
        stopping_patience=50
    )

    # Initiate our model
    wf_model = WalkForwardMLPPredictor(model=mlp_model, start_date="2019-01-01", end_date="2020-10-13",
                                       input_pct_change=1, output_pct_change=1, window_size=252, frequency=7,
                                       prediction_length=10, validation_size=21, sliding_window=True,
                                       random_validation=False)

    # Train our model through time, and obtain the predictions and errors
    mlp_predictions, mlp_error = wf_model.train_and_predict(input_data, output_data)
    print("MLP Walk Forward")

    sav_dates = pd.DataFrame(mlp_error)
    sav_dates = sav_dates.reset_index()

    saved = pd.read_csv(r'C:/Users/ELNA SIMONIS/Documents/Results/TESTING.csv')
    saved = saved.drop(['Unnamed: 0'], axis=1)

    saved['Dates'] = sav_dates['Date']
    saved = saved.set_index('Dates')
    saved['error'] = saved['TRUE'] - saved['PRED']
    saved = saved.dropna()

    # Calculate RMSE
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from math import sqrt

    mse = mean_squared_error(saved['TRUE'], saved['PRED'])
    rmse = sqrt(mean_squared_error(saved['TRUE'], saved['PRED']))
    mae = mean_absolute_error(saved['TRUE'], saved['PRED'])

    # Create a plot of our errors through time

    plt.figure(figsize=(10, 5))
    figuur = saved['error'] ** 2.0
    figuur.plot(color='blue')
    plt.xlabel('Dates', fontsize=15, fontweight='bold', color='black')
    plt.ylabel('Error', fontsize=15, fontweight='bold', color='black')
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.show()
    plt.close()