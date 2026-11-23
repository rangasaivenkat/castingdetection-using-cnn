
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


def get_predictions(model, dataset):
    """Get actual labels and predicted probabilities from a dataset."""

    true_labels = []
    probabilities = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0).ravel()

        probabilities.extend(predictions)
        true_labels.extend(labels.numpy().ravel())

    true_labels = np.array(true_labels).astype(int)
    probabilities = np.array(probabilities)

    return true_labels, probabilities


def find_best_threshold(true_labels, probabilities):
    """Find the threshold that gives the highest F1-score."""

    thresholds = np.arange(0.1, 0.91, 0.01)

    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        predicted_labels = (probabilities >= threshold).astype(int)

        score = f1_score(
            true_labels,
            predicted_labels,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    print("\nValidation threshold tuning")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best validation F1-score: {best_f1:.4f}")

    return best_threshold


def evaluate_model(model, val_data, test_data, class_names):
    """Evaluate the model using validation and test datasets."""

    # Tune the classification threshold using validation data
    val_labels, val_probabilities = get_predictions(
        model,
        val_data
    )

    threshold = find_best_threshold(
        val_labels,
        val_probabilities
    )

    # Evaluate the model on the test dataset
    true_labels, probabilities = get_predictions(
        model,
        test_data
    )

    predicted_labels = (probabilities >= threshold).astype(int)

    print("\nTest Results")
    print(f"Threshold : {threshold:.2f}")

    print(f"Accuracy  : {accuracy_score(true_labels, predicted_labels):.4f}")

    print(f"Precision : {precision_score(true_labels, predicted_labels, zero_division=0):.4f}")

    print(f"Recall    : {recall_score(true_labels, predicted_labels, zero_division=0):.4f}")

    print(f"F1-score  : {f1_score(true_labels, predicted_labels, zero_division=0):.4f}")

    print(f"ROC-AUC   : {roc_auc_score(true_labels, probabilities):.4f}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    print("\nConfusion Matrix")
    print(cm)

    # Classification report
    print("\nClassification Report")

    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=[0, 1],
            target_names=class_names,
            zero_division=0
        )
    )

    # Check prediction probabilities
    print("\nProbability Statistics")
    print(f"Min : {probabilities.min():.6f}")
    print(f"Max : {probabilities.max():.6f}")
    print(f"Mean: {probabilities.mean():.6f}")

    # Visualize model performance
    plot_confusion_matrix(cm, class_names)
    plot_roc_curve(true_labels, probabilities)

    return true_labels, probabilities, predicted_labels


def plot_confusion_matrix(cm, class_names):
    """Plot the confusion matrix."""

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=45
    )

    plt.yticks(
        range(len(class_names)),
        class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Display the number of predictions in each cell
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(col,row,
                cm[row, col],
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.show()


def plot_roc_curve(true_labels, probabilities):
    """Plot the ROC curve."""

    fpr, tpr, _ = roc_curve(
        true_labels,
        probabilities
    )

    auc_score = roc_auc_score(
        true_labels,
        probabilities
    )

    plt.figure(figsize=(6, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_score:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
