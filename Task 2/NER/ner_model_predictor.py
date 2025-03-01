from simpletransformers.ner import NERModel


class NERModelPredictor:
    def __init__(self, model_path):
        self.model = NERModel("bert", model_path)

    def predict(self, input_text):
        """Make predictions using the trained model."""
        predictions, _ = self.model.predict([input_text])
        return predictions
