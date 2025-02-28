import json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from models_interface import MnistClassifierInterface
from sklearn.model_selection import train_test_split


class NNClassifier(MnistClassifierInterface):
    def __init__(self, config_path):
        with open(config_path, "r") as file:
            config = json.load(file)["NNClassifier"]

        input_shape = tuple(config["input_shape"])
        self.num_classes = config["output_units"]

        self.model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(config["dense_layers"][0]["units"], activation=config["dense_layers"][0]["activation"]),
            layers.Dropout(config["dense_layers"][0]["dropout"]),
            layers.Dense(config["dense_layers"][1]["units"], activation=config["dense_layers"][1]["activation"]),
            layers.Dropout(config["dense_layers"][1]["dropout"]),
            layers.Dense(self.num_classes, activation=config["output_activation"])
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config["optimizer"]["learning_rate"]),
            loss=config["loss"],
            metrics=config["metrics"]
        )

        self.val_size = config["train_params"]["val_size"]
        self.epochs = config["train_params"]["epochs"]
        self.batch_size = config["train_params"]["batch_size"]

    def train(self, X_train, y_train):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train.reshape(X_train.shape[0], -1), to_categorical(y_train, num_classes=self.num_classes), test_size=self.val_size, random_state=42
        )

        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, X):
        predictions = self.model.predict(X.reshape(X.shape[0], -1))
        return np.argmax(predictions, axis=1)
