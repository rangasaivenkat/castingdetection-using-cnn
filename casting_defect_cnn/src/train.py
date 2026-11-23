
from pathlib import Path

import numpy as np
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight


MODEL_PATH = Path("models/casting_cnn.keras")


def calculate_class_weights(train_data):
    """Calculate class weights to handle imbalanced training data."""

    labels = []

    for _, batch_labels in train_data:
        labels.extend(batch_labels.numpy().ravel())

    labels = np.array(labels).astype(int)
    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }

    print("\nClass weights:")
    print(class_weights)

    return class_weights


def train_model(model, train_data, val_data, epochs=20):
    """Train the CNN and save the best performing model."""

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    class_weights = calculate_class_weights(train_data)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),

        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )

    model.save(MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")

    return history
