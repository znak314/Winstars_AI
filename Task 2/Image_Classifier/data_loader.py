import pandas as pd
import os
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split


class ImageDataPipeline:
    def __init__(self, dataset_path, test_size=0.15, batch_size=16, target_size=(224, 224)):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.batch_size = batch_size
        self.target_size = target_size
        self.train_df, self.test_df = self._load_dataset()
        self.train_generator, self.test_generator = self._init_generators()
        self.train_images, self.validation_images, self.test_images = self._prepare_data()
        self.class_indices = self.train_images.class_indices

    def _load_dataset(self):
        """Converts image dataset into a DataFrame."""
        extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
        filepaths = [str(path) for ext in extensions for path in Path(self.dataset_path).rglob(f"*.{ext}")]
        labels = [os.path.basename(os.path.dirname(path)) for path in filepaths]
        df = pd.DataFrame({"filepath": filepaths, "label": labels})
        return train_test_split(df, test_size=self.test_size, shuffle=True, random_state=42)

    def _init_generators(self):
        """Initializes ImageDataGenerators."""
        train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
            validation_split=0.2,
            rescale=1.0 / 255,
        )

        test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
            rescale=1.0 / 255,
        )

        return train_gen, test_gen

    def _prepare_data(self):
        """Creates data generators for training, validation, and testing."""
        train_images = self.train_generator.flow_from_dataframe(
            self.train_df,
            x_col='filepath',
            y_col='label',
            target_size=self.target_size,
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            subset='training',
            seed=42
        )

        validation_images = self.train_generator.flow_from_dataframe(
            self.train_df,
            x_col='filepath',
            y_col='label',
            target_size=self.target_size,
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            subset='validation',
            seed=42
        )

        test_images = self.test_generator.flow_from_dataframe(
            self.test_df,
            x_col='filepath',
            y_col='label',
            target_size=self.target_size,
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )

        return train_images, validation_images, test_images
