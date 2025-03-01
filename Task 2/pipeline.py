import tensorflow as tf
from NER.ner_model_predictor import NERModelPredictor
from Image_Classifier.inference import predict

class Pipeline:
    def __init__(self, NER_model_path, classifier_model_path):
        self.NER_model = NERModelPredictor(NER_model_path)
        self.classifier_model = tf.keras.models.load_model(classifier_model_path)

    def predict_animal_on_image(self, image_path, text):
        prediction = self.NER_model.predict(text)
        if not prediction:
            return False

        # Get animal tokens
        animal_tokens = [list(token.keys())[0] for word in prediction for token in word if
                         list(token.values())[0] == 'ANIMAL']

        if not animal_tokens:
            print("No animals in text")
            return False

        predicted_class = predict(self.classifier_model, image_path)
        return predicted_class in animal_tokens
