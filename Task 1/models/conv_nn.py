import json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from models_interface import MnistClassifierInterface


class CNNClassifier(MnistClassifierInterface):
    def __init__(self, config_path):

        with open(config_path, "r") as f:
            all_configs = json.load(f)

        self.config = all_configs["CNNClassifier"]
        self.batch_size = self.config["batch_size"]

        self.model = keras.Sequential()

        for i, conv_layer in enumerate(self.config["conv_layers"]):
            if i == 0:
                self.model.add(
                    layers.Conv2D(
                        filters=conv_layer["filters"],
                        kernel_size=tuple(conv_layer["kernel_size"]),
                        activation=conv_layer["activation"],
                        padding=conv_layer["padding"],
                        input_shape=tuple(self.config["input_shape"])
                    )
                )
            else:
                self.model.add(
                    layers.Conv2D(
                        filters=conv_layer["filters"],
                        kernel_size=tuple(conv_layer["kernel_size"]),
                        activation=conv_layer["activation"],
                        padding=conv_layer["padding"]
                    )
                )
            self.model.add(layers.MaxPooling2D(pool_size=tuple(self.config["pool_size"])))

        # Додавання повнозв'язних шарів
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.config["dense_units"], activation=self.config["activation_dense"]))
        self.model.add(layers.Dense(self.config["output_units"], activation=self.config["output_activation"]))

        # Компляція моделі
        self.model.compile(
            optimizer=self.config["optimizer"],
            loss=self.config["loss"],
            metrics=self.config["metrics"]
        )

    def train(self, X_train, y_train):
        X_train = X_train.reshape(-1, *self.config["input_shape"])
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.config["val_size"], random_state=42
        )

        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config["epochs"],
            batch_size=self.batch_size
        )

    def predict(self, X_test):
        X_test = X_test.reshape(-1, *self.config["input_shape"])
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)


'''
    def predict(self, X_test):
        predictions = self.model.predict(X_test.reshape(-1, * self.config["input_shape"]))
        return np.argmax(predictions, axis=1)
'''