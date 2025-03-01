import argparse
from data_loader import ImageDataPipeline
from animal_classifier import AnimalClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    # Load data
    data_pipeline = ImageDataPipeline(args.dataset_path, batch_size=args.batch_size)

    # Initialize and train model
    classifier = AnimalClassifier(num_classes=len(data_pipeline.class_indices))
    classifier.compile_model()
    classifier.train(
        data_pipeline.train_images, data_pipeline.validation_images, epochs=args.epochs
    )

    # Evaluate on test data
    classifier.evaluate(data_pipeline.test_images)

    # Save the trained model
    classifier.save_model("animal_classifier.h5")

if __name__ == "__main__":
    main()