import argparse
from ner_model_predictor import NERModelPredictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--input_text", type=str, required=True, help="Input text for NER inference")

    args = parser.parse_args()

    # Make predictions
    predictor = NERModelPredictor(args.model_path)
    predictions = predictor.predict(args.input_text)

    print("Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()
