import argparse
from data_loader import DataLoader
from ner_model_trainer import NERModelTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--output_dir", type=str, default="trained_ner_model", help="Directory to save the trained model")

    args = parser.parse_args()

    # Load and prepare data
    data_loader = DataLoader(args.file_path)
    train_data, test_data = data_loader.prepare_data()

    # Train the model with dynamic parameters
    model_trainer = NERModelTrainer(
        train_data,
        test_data,
        labels=data_loader.data["label"].unique().tolist(),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    model_trainer.train()
    evaluation_result = model_trainer.evaluate()
    print(f"Evaluation Results: {evaluation_result}")
    model_trainer.save()

if __name__ == "__main__":
    main()
