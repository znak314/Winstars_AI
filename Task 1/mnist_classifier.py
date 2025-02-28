from models.random_forest import RandomForestModel
from models.feed_forward_nn import NNClassifier
from models.conv_nn import CNNClassifier

class MnistClassifier:
    def __init__(self, algorithm, config_path):
        if algorithm == "rf":
            self.model = RandomForestModel(config_path)
        elif algorithm == "nn":
            self.model = NNClassifier(config_path)
        elif algorithm == "cnn":
            self.model = CNNClassifier(config_path)
        else:
            raise ValueError("Invalid algorithm. Choose 'rf', 'nn', or 'cnn'.")

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)