import json
from sklearn.ensemble import RandomForestClassifier
from models_interface import MnistClassifierInterface


class RandomForestModel(MnistClassifierInterface):
    "Random Forest classifier"
    def __init__(self, config_path):

        with open(config_path, "r") as file:
            config = json.load(file)["RandomForestModel"]

        self.model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=config["random_state"],
            min_samples_split=config.get("min_samples_split", 2),
            min_samples_leaf=config.get("min_samples_leaf", 1),
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)