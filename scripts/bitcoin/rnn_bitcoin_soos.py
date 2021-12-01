import warnings
# This is hacky, but due to tensorflow requiring lower numpy version, but pmdarima requiring higher version, this is
# done to clear the console.
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.models.rnn import RecurrentNN

from src.utils import series_to_supervised

# TODO: Add description! Mention datasources

# =====================================================================================================================
# GET DATA
# =====================================================================================================================

# Set number of features
n_features = 2

# Get data path or create a directory if it does not exist
# TODO: This is hacky. Need to fix
pathlib.Path(os.path.join(os.path.dirname(os.getcwd()), "..", "data")).mkdir(parents=True, exist_ok=True)
data_path = os.path.join(os.path.dirname(os.getcwd()), "..", "data")

# Check if file exists
if not os.path.exists(os.path.join(data_path, "bitcoin.csv")):
    raise ValueError("No data in data folder!")

# Get bitcoin data
bitcoin_data = pd.read_csv(os.path.join(data_path, "local_etfs_close.csv"), index_col=0)
bitcoin_data = bitcoin_data.to_frame().ffill().dropna()

# Make data stationary
bitcoin_data = bitcoin_data.pct_change()

# Create supervised learning problem
bitcoin_data = series_to_supervised(bitcoin_data, n_in=n_features, n_out=1)

# Create training and testing data
x_train, x_test, y_train, y_test = train_test_split(bitcoin_data.iloc[:, :-1], bitcoin_data.iloc[:, -1],
                                                    test_size=0.1, random_state=1, shuffle=False)

# Create validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.01, random_state=1, shuffle=False)

print("x_train: ", np.shape(x_train))
print("x_val: ", np.shape(x_val))
print("y_train: ", np.shape(y_train))
print("y_val: ", np.shape(y_val))

# =====================================================================================================================
# BUILD MODEL
# =====================================================================================================================


# Create RNN
rnn = RecurrentNN(
    name="rnn_bitcoin_nwf",
    num_inputs=n_features,
    num_outputs=1,
    # If true, training info is outputted to stdout
    keras_verbose=True,
    # A summary of the NN is printed to stdout
    print_model_summary=True,
    # rnn_layers = [units, kernel_regularizer (l2), recurrent_regularizer (l2), dropout, recurrent_dropout]
    rnn_layers=[
        [50, 0.0, 0.0, 0.2, 0.0],
        [50, 0.0, 0.0, 0.2, 0.0],
        [50, 0.0, 0.0, 0.2, 0.0]
    ],
    # Statefulness
    stateful_training=False,
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
    epochs=2000,
    # The batch size to use in the NN
    batch_size=32,
    # The learning rate used in optimization
    learning_rate=0.001,
    # If this many stagnant epochs are seen, stop training
    stopping_patience=15
)

# Train MLP model from scratch
rnn.train(x_train, y_train, x_val, y_val, load_model=False)

# Predict
predicted = rnn.predict(x_test)

# Create dataframe for predicted values
pred_df = pd.DataFrame(np.column_stack([np.squeeze(predicted), y_test]))
pred_df.columns = ["PRED", "TRUE"]

# Plot predicted values
pred_df.plot()
plt.show()
plt.close()
