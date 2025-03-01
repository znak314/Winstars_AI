import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from data_loader import ImageDataPipeline


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess the image for inference."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array


def predict(model, img_path):
    """Predict the class label and confidence of the given image."""
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    class_labels = [
        'butterfly',
        'cat',
        'chicken',
        'cow',
        'dog',
        'elephant',
        'horse',
        'sheep',
        'spider',
        'squirrel'
    ]

    return class_labels[predicted_class]


def main():
    """Main function for handling command-line inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, default="animal_classifier.h5", help="Path to trained model")
    args = parser.parse_args()

    # Load the trained model only once
    model = tf.keras.models.load_model(args.model_path)

    # Make prediction
    predicted_label = predict(model, args.img_path)

    print(f"Predicted Class: {predicted_label}")


if __name__ == "__main__":
    main()
