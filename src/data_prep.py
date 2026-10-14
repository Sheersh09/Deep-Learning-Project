"""
Step 1: Explore the dataset and create train/val/test splits.

Expected input layout:

    data/raw/
        Tomato___healthy/
        Tomato___Early_blight/
        Tomato___Late_blight/

This script:
  1. Verifies the three class folders exist.
  2. Counts images in each class.
  3. Displays sample images.
  4. Checks image dimensions.
  5. Creates an 80/10/10 train/validation/test split.

Run this once before training.
"""

import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.image import imread
import splitfolders


RAW_DIR = "data/raw"
SPLIT_DIR = "data/split"
FIGURES_DIR = "outputs/figures"

CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SEED = 42


def verify_dataset():
    """Check that all expected class folders exist."""

    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(f"'{RAW_DIR}' not found.")

    missing = [
        class_name
        for class_name in CLASSES
        if not os.path.isdir(os.path.join(RAW_DIR, class_name))
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing class folder(s): {missing}"
        )

    print("Dataset verified.")
    for class_name in CLASSES:
        print(f"  ✓ {class_name}")


def get_image_files(folder):
    """Return only image files from a folder."""

    return [
        filename
        for filename in os.listdir(folder)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    ]


def count_images_per_class():
    """Count and plot images in each class."""

    counts = {}

    for class_name in CLASSES:
        folder = os.path.join(RAW_DIR, class_name)
        counts[class_name] = len(get_image_files(folder))

    print("\nImage counts:")

    for class_name, count in counts.items():
        print(f"  {class_name}: {count}")

    total = sum(counts.values())
    print(f"  Total: {total}")

    plt.figure(figsize=(7, 4))
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.title("Number of Images per Class")
    plt.ylabel("Image Count")
    plt.xticks(rotation=20)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(
        os.path.join(FIGURES_DIR, "class_distribution.png")
    )

    plt.show()

    return counts


def show_example_images(n_per_class=3):
    """Display a few example images from each class."""

    fig, axes = plt.subplots(
        len(CLASSES),
        n_per_class,
        figsize=(n_per_class * 3, len(CLASSES) * 3),
    )

    for row, class_name in enumerate(CLASSES):

        folder = os.path.join(RAW_DIR, class_name)
        filenames = get_image_files(folder)[:n_per_class]

        for col, filename in enumerate(filenames):

            image_path = os.path.join(folder, filename)
            image = imread(image_path)

            ax = axes[row, col]
            ax.imshow(image)
            ax.axis("off")

            if col == 0:
                ax.set_title(class_name, fontsize=9)

    plt.suptitle("Example Images from Each Class")
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURES_DIR, "example_images.png")
    )

    plt.show()


def check_image_dimensions(n_per_class=5):
    """Check dimensions of sample images."""

    print("\nChecking image dimensions...\n")

    all_shapes = []

    for class_name in CLASSES:

        folder = os.path.join(RAW_DIR, class_name)
        filenames = get_image_files(folder)[:n_per_class]

        for filename in filenames:

            image_path = os.path.join(folder, filename)
            image = imread(image_path)

            all_shapes.append(image.shape)

            print(
                f"{class_name}/{filename}: {image.shape}"
            )

    unique_shapes = set(all_shapes)

    print(f"\nUnique shapes found: {unique_shapes}")

    if len(unique_shapes) == 1:
        print("All sampled images have the same dimensions.")
    else:
        print(
            "Images have different dimensions. "
            "They will later be resized to 128x128."
        )


def split_dataset():
    """Create an 80/10/10 train/validation/test split."""

    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    splitfolders.ratio(
        RAW_DIR,
        output=SPLIT_DIR,
        seed=SEED,
        ratio=(0.8, 0.1, 0.1),
    )

    print("\nSplit complete.")
    print(f"Training/validation/test data saved in: {SPLIT_DIR}")


if __name__ == "__main__":

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("STEP 1: VERIFY DATASET")
    verify_dataset()

    print("\nSTEP 2: COUNT IMAGES")
    count_images_per_class()

    print("\nSTEP 3: SHOW EXAMPLE IMAGES")
    show_example_images()

    print("\nSTEP 4: CHECK IMAGE DIMENSIONS")
    check_image_dimensions()

    print("\nSTEP 5: CREATE TRAIN/VAL/TEST SPLIT")
    split_dataset()