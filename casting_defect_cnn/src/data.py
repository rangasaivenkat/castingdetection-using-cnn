
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Dataset configuration
DATASET_PATH = Path("/Users/saivenkatr/Downloads/archive/casting_data/casting_data")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.2

CLASS_DIRS = ["ok_front", "def_front"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

AUTOTUNE = tf.data.AUTOTUNE


def get_image_paths(directory):
    """Get image paths and labels from the class folders."""

    image_paths = []
    labels = []

    for label, class_name in enumerate(CLASS_DIRS):
        class_dir = Path(directory) / class_name

        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(image_path))
                labels.append(label)

    return np.array(image_paths), np.array(labels, dtype=np.int32)


def print_class_counts(labels, split_name):
    """Print the number of images in each class."""

    print(f"\n{split_name} class distribution:")

    for label, class_name in enumerate(CLASS_DIRS):
        count = np.sum(labels == label)
        print(f"{class_name}: {count}")


def load_image(image_path, label):
    """Load, resize and normalize an image."""

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=1,
        expand_animations=False
    )

    image.set_shape([None, None, 1])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0

    label = tf.cast(label, tf.float32)

    return image, label


def create_dataset(image_paths, labels, training=False):
    """Create a TensorFlow dataset."""

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            len(image_paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def load_datasets():
    """Load local data and create train, validation and test datasets."""

    train_dir = DATASET_PATH / "train"
    test_dir = DATASET_PATH / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Check the dataset directory."
        )

    # Load training images
    image_paths, labels = get_image_paths(train_dir)

    print(f"Total training images: {len(image_paths)}")

    # Split training data into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=labels
    )

    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")

    print_class_counts(train_labels, "Training")
    print_class_counts(val_labels, "Validation")

    # Load test images
    test_paths, test_labels = get_image_paths(test_dir)

    print(f"\nTotal testing images: {len(test_paths)}")
    print_class_counts(test_labels, "Testing")

    # Create datasets
    train_dataset = create_dataset(
        train_paths,
        train_labels,
        training=True
    )

    val_dataset = create_dataset(
        val_paths,
        val_labels
    )

    test_dataset = create_dataset(
        test_paths,
        test_labels
    )

    print("\nClass mapping:")
    for label, class_name in enumerate(CLASS_DIRS):
        print(f"{label}: {class_name}")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        CLASS_DIRS,
        test_paths
    )
