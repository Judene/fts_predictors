import numpy as np
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, GaussianDropout, SimpleRNN, CuDNNRNN, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

from src.models.fts_model import FTSModel

import src.utils as utils


class RecurrentNN(FTSModel):
    """
    An implementation of a RNN (Recurrent neural network).
    """

    def __init__(self, name: str, num_inputs: int, num_outputs: int, *args, **kwargs):
        """
        :param name: the user specified name given for the neural network
        :param num_inputs: the number of inputs that goes into the model
        :param num_outputs: the number of possible classes
        """
        self.name = name
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # If true, training info is outputted to stdout
        self.keras_verbose = False
        # A summary of the NN is printed to stdout
        self.print_model_summary = False

        # rnn_layers = [units, kernel_regularizer (l2), recurrent_regularizer (l2), dropout, recurrent_dropout]
        self.rnn_layers = [
            [50, 0.0, 0.0, 0.0, 0.0],
            [50, 0.0, 0.0, 0.0, 0.0],
            [50, 0.0, 0.0, 0.0, 0.0]
        ]

        # Whether to use CUDNN layers
        self.use_cudnn_layers = False

        # Statefulness
        self.stateful_training = False

        # ff_layers = [units, activation, regularization, dropout, use_bias, dropout_type]
        self.ff_layers = [
            [512, "relu", 0.0, 0.2, True, "gaussian"],
            [512, "relu", 0.0, 0.2, True, "gaussian"],
            [512, "relu", 0.0, 0.2, True, "gaussian"]
        ]

        # The final output layer's activation function
        self.final_activation = "tanh"
        # The objective function for the NN
        self.objective = "mse"
        # The maximum number of epochs to run
        self.epochs = 20
        # The batch size to use in the NN
        self.batch_size = 128
        # The learning rate used in optimization
        self.learning_rate = 0.001
        # If this many stagnant epochs are seen, stop training
        self.stopping_patience = 20

        # If the model is stateful then force the batch size to 1
        if self.stateful_training:
            self.batch_size = 1
            print("Stateful training is enabled. "
                  "Setting self.batch_size=1")

        self.__dict__.update(kwargs)

        # The optimization algorithm to use
        self.optimizer = Adam(lr=self.learning_rate)

        # --------------------------------------------------------------------------------

        # Create the input
        inputs = [
            Input(
                shape=(num_inputs,)
            )
        ]

        # Create the layers
        layers = []
        layers.extend(inputs)

        # For RNN, we need 3D inputs (num_samples, num_timesteps, num_features)
        layers.append(
            Reshape((1, num_inputs))(layers[-1])
        )

        for i in range(len(self.rnn_layers)):

            # Determine if we should return sequences
            return_sequences = i != (len(self.rnn_layers) - 1)

            # --- Add a dropout layer on the inputs ---

            layers.append(
                Dropout(
                    rate=self.rnn_layers[i][3]
                )(layers[-1])
            )

            # --- Add a Gated Recurrent Unit layer to the network ---

            if self.use_cudnn_layers:
                layers.append(
                    CuDNNRNN(
                        units=self.rnn_layers[i][0],
                        stateful=self.stateful_training,
                        return_sequences=return_sequences,
                        kernel_regularizer=l2(self.rnn_layers[i][1]),
                        recurrent_regularizer=l2(self.rnn_layers[i][2])
                    )(layers[-1])
                )
            else:
                layers.append(
                    SimpleRNN(
                        units=self.rnn_layers[i][0],
                        stateful=self.stateful_training,
                        return_sequences=return_sequences,
                        recurrent_dropout=self.rnn_layers[i][4],
                        kernel_regularizer=l2(self.rnn_layers[i][1]),
                        recurrent_regularizer=l2(self.rnn_layers[i][2])
                    )(layers[-1])
                )

        # Build up the layers
        for j in range(len(self.ff_layers)):

            # --- Add the Dense layers (fully connected) to the model ---

            layers.append(
                Dense(
                    units=self.ff_layers[j][0],
                    activation=self.ff_layers[j][1],
                    kernel_regularizer=l2(self.ff_layers[j][2]),
                    use_bias=self.ff_layers[j][4]
                )(layers[-1])
            )

            # --- Add the Dropout layers (normal or gaussian) to the model ---

            if self.ff_layers[j][5] == "normal":
                layers.append(
                    Dropout(
                        self.ff_layers[j][3]
                    )(layers[-1])
                )
            elif self.ff_layers[j][5] == "gaussian":
                layers.append(
                    GaussianDropout(
                        self.ff_layers[j][3]
                    )(layers[-1])
                )

        # Add the final dense layer
        output = Dense(
            units=self.num_outputs,
            activation=self.final_activation
        )(layers[-1])

        # Create the neural network model and compile it
        nnet = Model(inputs=inputs, outputs=output, name=self.name)

        # Finally compile the neural network so we can use it with the weights going forward
        nnet.compile(optimizer=self.optimizer, loss=self.objective)

        # Call the parent class constructor to initialize the object's variables
        super().__init__(name, nnet, *args, **kwargs)

        # Assign nnet to self.model
        self.model = nnet
        if self.print_model_summary:
            print("")
            self.model.summary()

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray,
              load_model=True) -> None:

        if load_model:
            self.load()

        else:
            # What we want back from Keras
            callbacks_list = []

            # The default patience is stopping_patience
            patience = self.stopping_patience

            # Create an early stopping callback and add it
            callbacks_list.append(
                EarlyStopping(
                    verbose=1,
                    monitor="val_loss",
                    patience=patience))

            # Train the model
            training_process = self.model.fit(
                x=x_train,
                y=y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.keras_verbose,
                callbacks=callbacks_list,
                validation_data=(x_valid, y_valid)
            )

            self.save()

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> tuple:
        score = self.model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=0)
        test_loss = score[0]
        test_accuracy = score[1]
        print("accuracy: {:.2f}% | loss: {}".format(100 * test_accuracy, test_loss))
        print("SCORE: ", score)
        return test_loss, test_accuracy

    def save(self):
        project_dir = Path(__file__).resolve().parents[2]
        models_dir = str(project_dir) + '/models/' + self.name + '/'
        utils.check_folder(models_dir)
        self.model.save(models_dir + self.name + ".h5")
        print("Saved model to disk")

    def load(self):
        try:
            project_dir = Path(__file__).resolve().parents[2]
            models_dir = str(project_dir) + '/models/' + self.name + '/'
            utils.check_folder(models_dir)
            self.model = load_model(models_dir + self.name + ".h5")
            print("Loaded " + self.name + " model from disk")

        except ValueError as e:
            print("No saved model found. Check file name or train from scratch")

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(x)
        return prediction

    def plot_model(self):
        plot_model(self.model, to_file=self.name + "_model.png", show_shapes=True)
