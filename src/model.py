"""
CNN architecture:

    Input -> Conv2D -> ReLU -> MaxPool
          -> Conv2D -> ReLU -> MaxPool
          -> Flatten -> Dense -> ReLU
          -> Dropout -> Dense -> Softmax

This is a small CNN trained from scratch so that each layer
and its purpose can be clearly understood.
"""

from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore


IMG_SIZE = 128
NUM_CLASSES = 3
DROPOUT_RATE = 0.5


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # First convolution block
        layers.Conv2D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        # Second convolution block
        layers.Conv2D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        # Convert feature maps into a 1D vector
        layers.Flatten(),

        # Fully connected layer
        layers.Dense(128, activation="relu"),

        # Regularization to reduce overfitting
        layers.Dropout(DROPOUT_RATE),

        # Output: probability for each of the 3 classes
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()