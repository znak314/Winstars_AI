from simpletransformers.ner import NERModel, NERArgs

class NERModelTrainer:
    def __init__(self, train_data, test_data, labels, epochs=3, learning_rate=2e-4, batch_size=16, output_dir="trained_ner_model"):
        self.train_data = train_data
        self.test_data = test_data
        self.labels = labels
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.model = self.create_model()

    def create_model(self):
        """Create and return a NER model."""
        args = NERArgs(
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            overwrite_output_dir=True,
            output_dir=self.output_dir
        )

        return NERModel('bert', 'bert-base-cased', labels=self.labels, args=args, use_cuda=False)

    def train(self):
        """Train the NER model."""
        self.model.train_model(self.train_data, eval_data=self.test_data)

    def evaluate(self):
        """Evaluate the trained model."""
        result, model_outputs, preds_list = self.model.eval_model(self.test_data)
        return result

    def save(self):
        """Save the trained model."""
        self.model.save_model(self.output_dir)