"""
Step 4: Evaluate the trained CNN models.

This script:
1. Loads the baseline CNN and augmented CNN.
2. Evaluates both on the unseen test set.
3. Calculates accuracy, precision, recall, and F1-score.
4. Creates a comparison table.
5. Creates a confusion matrix for the augmented model.
6. Displays some misclassified test images.

Run from the project root:

    python src/evaluate.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Any, cast
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

from sklearn.metrics import (classification_report,confusion_matrix,ConfusionMatrixDisplay,)
from model import IMG_SIZE


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SPLIT_DIR = "data/split"
MODELS_DIR = "outputs/models"
RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/figures"

BATCH_SIZE = 32

CLASS_NAMES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]


# --------------------------------------------------
# Test data
# --------------------------------------------------

def make_test_generator():
    """Create the test data generator."""

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    return test_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode="categorical",
        shuffle=False,
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_model(model_path):
    """Evaluate a saved model on the test set."""

    test_gen = make_test_generator()
    model = load_model(model_path)

    # Evaluate loss and accuracy
    test_loss, test_accuracy = model.evaluate(
        test_gen,
        verbose=1
    )

    # Generate predictions
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=0)

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    true_labels = test_gen.classes

    # Classification report
    report = cast(
    dict[str, Any],
    classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASS_NAMES,
        output_dict=True,
    ),
)

    print(f"\n{'=' * 60}")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"{'=' * 60}")

    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=CLASS_NAMES,
        )
    )

    metrics = {
        "Accuracy": report["accuracy"],
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1": report["weighted avg"]["f1-score"],
    }

    return (
        true_labels,
        predicted_labels,
        metrics,
    )


# --------------------------------------------------
# Model comparison
# --------------------------------------------------

def build_comparison_table(
    baseline_metrics,
    augmented_metrics
):
    """Create and save a comparison table."""

    table = pd.DataFrame([
        {
            "Model": "Baseline CNN",
            **baseline_metrics,
        },
        {
            "Model": "CNN + Augmentation",
            **augmented_metrics,
        },
    ])

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]

    table[metric_columns] = table[
        metric_columns
    ].round(4)

    print("\nModel Comparison:\n")
    print(table.to_string(index=False))

    table.to_csv(
        os.path.join(
            RESULTS_DIR,
            "model_comparison.csv"
        ),
        index=False,
    )

    return table


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

def plot_confusion_matrix(
    true_labels,
    predicted_labels,
):
    """Create a confusion matrix for the augmented model."""

    display_names = [
        "Early Blight",
        "Late Blight",
        "Healthy",
    ]

    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_names,
    )

    display.plot(
        xticks_rotation=20
    )

    plt.title(
        "Confusion Matrix - Augmented CNN"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "confusion_matrix.png"
        )
    )

    plt.show()


# --------------------------------------------------
# Load test images
# --------------------------------------------------

def load_test_images():
    """Load all test images in generator order."""

    test_gen = make_test_generator()

    images = []

    for _ in range(len(test_gen)):
        batch_images, _ = next(test_gen)
        images.append(batch_images)

    return np.concatenate(images, axis=0)


# --------------------------------------------------
# Misclassified images
# --------------------------------------------------

def show_misclassified(
    true_labels,
    predicted_labels,
    n=8,
):
    """Display some incorrectly classified test images."""

    wrong_indices = np.where(
        true_labels != predicted_labels
    )[0]

    if len(wrong_indices) == 0:
        print("No misclassified test images.")
        return

    wrong_indices = wrong_indices[:n]

    test_images = load_test_images()

    display_names = [
        "Early Blight",
        "Late Blight",
        "Healthy",
    ]

    fig, axes = plt.subplots(
        1,
        len(wrong_indices),
        figsize=(len(wrong_indices) * 2.5, 3),
    )

    if len(wrong_indices) == 1:
        axes = [axes]

    for ax, index in zip(
        axes,
        wrong_indices
    ):

        ax.imshow(test_images[index])

        ax.set_title(
            f"Actual: {display_names[true_labels[index]]}\n"
            f"Predicted: {display_names[predicted_labels[index]]}",
            fontsize=8,
        )

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "misclassified_examples.png"
        )
    )

    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        FIGURES_DIR,
        exist_ok=True
    )

    print("Evaluating baseline CNN...")

    (
        baseline_true,
        baseline_pred,
        baseline_metrics,
    ) = evaluate_model(
        os.path.join(
            MODELS_DIR,
            "model_baseline.keras"
        )
    )

    print("\nEvaluating augmented CNN...")

    (
        augmented_true,
        augmented_pred,
        augmented_metrics,
    ) = evaluate_model(
        os.path.join(
            MODELS_DIR,
            "model_augmented.keras"
        )
    )

    build_comparison_table(
        baseline_metrics,
        augmented_metrics,
    )

    plot_confusion_matrix(
        augmented_true,
        augmented_pred,
    )

    show_misclassified(
        augmented_true,
        augmented_pred,
        n=8,
    )

    print("\nEvaluation complete.")