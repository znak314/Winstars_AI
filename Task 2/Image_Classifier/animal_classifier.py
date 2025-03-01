import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


class AnimalClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10, learning_rate=1e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        base_model = tf.keras.applications.EfficientNetB6(
            include_top=False, weights='imagenet', input_shape=self.input_shape, pooling='max'
        )
        for layer in base_model.layers:
            layer.trainable = False

        model = Sequential([
            base_model,
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            Dense(self.num_classes, activation='softmax')
        ])
        return model

    def compile_model(self):
        self.model.compile(
            optimizer=Adam(self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_data, val_data, epochs=10, callbacks=[]):
        return self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )

    def evaluate(self, test_data):
        loss, accuracy = self.model.evaluate(test_data)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    def save_model(self, path='animal_classifier.h5'):
        self.model.save(path)