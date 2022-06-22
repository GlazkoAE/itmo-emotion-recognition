"""
A collection of models we'll use to attempt to classify videos.
"""
import sys
from collections import deque

from keras.layers import LSTM, Dense, Dropout
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError, MeanSquaredLogarithmicError
from keras.models import Sequential, load_model
from keras.optimizers import Adam


class ResearchModels:
    def __init__(
        self, nb_classes, model, seq_length, saved_model=None, features_length=2048
    ):
        """
        `model` = only 'lstm' this moment
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics
        metrics = [MeanSquaredLogarithmicError(), MeanAbsoluteError()]

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == "lstm":
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(
            loss=MeanSquaredError(), optimizer=optimizer, metrics=metrics
        )

        print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predominantly."""
        # Model.
        model = Sequential()
        model.add(
            LSTM(
                2048, return_sequences=False, input_shape=self.input_shape, dropout=0.5
            )
        )
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))

        # for linear regression
        model.add(Dense(1, activation="linear"))

        return model
