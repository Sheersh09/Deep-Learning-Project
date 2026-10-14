"""
Train the baseline and augmented CNN models.

Model A:
    Baseline CNN without data augmentation

Model B:
    Same CNN with data augmentation during training

Both models use the same architecture, dataset split, and training settings. The purpose is to compare their performance.
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

from model import build_model, IMG_SIZE


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SPLIT_DIR = "data/split"
MODELS_DIR = "outputs/models"
RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/figures"

BATCH_SIZE = 32
MAX_EPOCHS = 20
SEED = 42

CLASS_NAMES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]


# Make training more reproducible
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------
# Data generators
# --------------------------------------------------

def make_generators(augmented=False):
    """Create training and validation data generators."""

    if augmented:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255
        )

    # Validation data is never augmented.
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_one_model(augmented, run_name):
    """Train one CNN model and save the model and history."""

    print(f"\n--- Training {run_name} model ---")

    train_gen, val_gen = make_generators(augmented)

    print("\nClass mapping:")
    print(train_gen.class_indices)

    model = build_model()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=MAX_EPOCHS,
        callbacks=[early_stop],
    )

    model.save(
        os.path.join(
            MODELS_DIR,
            f"model_{run_name}.keras"
        )
    )

    pd.DataFrame(history.history).to_csv(
        os.path.join(
            RESULTS_DIR,
            f"{run_name}_history.csv"
        ),
        index=False,
    )

    return history


# --------------------------------------------------
# Comparison plot
# --------------------------------------------------

def plot_comparison(history_baseline, history_augmented):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(
        history_baseline.history["accuracy"],
        label="Baseline - Train"
    )
    axes[0].plot(
        history_baseline.history["val_accuracy"],
        label="Baseline - Validation"
    )
    axes[0].plot(
        history_augmented.history["accuracy"],
        label="Augmented - Train"
    )
    axes[0].plot(
        history_augmented.history["val_accuracy"],
        label="Augmented - Validation"
    )

    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(
        history_baseline.history["loss"],
        label="Baseline - Train"
    )
    axes[1].plot(
        history_baseline.history["val_loss"],
        label="Baseline - Validation"
    )
    axes[1].plot(
        history_augmented.history["loss"],
        label="Augmented - Train"
    )
    axes[1].plot(
        history_augmented.history["val_loss"],
        label="Augmented - Validation"
    )

    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "training_comparison.png"
        )
    )
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Training baseline CNN...")
    history_baseline = train_one_model(
        augmented=False,
        run_name="baseline"
    )

    print("\nTraining augmented CNN...")
    history_augmented = train_one_model(
        augmented=True,
        run_name="augmented"
    )

    print("\nCreating comparison plot...")
    plot_comparison(
        history_baseline,
        history_augmented
    )

    print("\nTraining complete.")