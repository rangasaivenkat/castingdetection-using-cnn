
from tensorflow import keras

from src.data import load_datasets
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    # Load the dataset
    train_data, val_data, test_data, class_names, test_files = load_datasets()

    # Build the CNN
    model = build_model()
    model.summary()

    # Train the model
    train_model(
        model,
        train_data,
        val_data,
        epochs=10
    )

    # Load the best saved model
    model = keras.models.load_model("models/casting_cnn.keras")

    # Evaluate on validation and test data
    evaluate_model(
        model,
        val_data,
        test_data,
        class_names
    )


if __name__ == "__main__":
    main()
