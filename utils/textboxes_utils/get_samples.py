import os
from glob import glob


def get_samples(images_dir, labels_dir):
    """ Create a list of samples that can be feed to a data generator.

    Args:
        - images_dir: Path to images directory.
        - labels_dir: Path to labels directory.

    Returns:
        - A list of samples. Each sample is a string containing paths to both the image file and its corresponding label file separated by space.

    Raises:
        - images_dir is not a directory.
        - labels_dir is not a directory.
    """
    assert os.path.isdir(images_dir), "images_dir is not a directory."
    assert os.path.isdir(labels_dir), "labels_dir is not a directory."

    images = glob(os.path.join(images_dir, "*.jpg"))
    labels = glob(os.path.join(images_dir, "*.txt"))

    print(len(images), len(labels))
