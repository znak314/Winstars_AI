import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()


    def load_data(self):
        """Load dataset from the given file path."""
        return pd.read_csv(self.file_path, encoding="utf-8")

    def prepare_data(self, test_size=0.15):
        """Prepare training and test datasets."""
        X = self.data[["sentence_id", "word", "POS"]]
        Y = self.data["label"]

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

        train_data = pd.DataFrame({
            "sentence_id": x_train["sentence_id"],
            "words": x_train["word"],
            "POS": x_train["POS"],
            "labels": y_train
        })

        test_data = pd.DataFrame({
            "sentence_id": x_test["sentence_id"],
            "words": x_test["word"],
            "POS": x_test["POS"],
            "labels": y_test
        })

        return train_data, test_data
