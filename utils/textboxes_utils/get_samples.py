import os
from glob import glob
from utils import textboxes_utils


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

    images = sorted(list(glob(os.path.join(images_dir, "*.jpg"))))
    labels = sorted(list(glob(os.path.join(labels_dir, "*.txt"))))

    assert len(images) == len(labels), "the number of images and the number of labels does not match"

    samples = []

    for (image_path, label_path) in list(zip(images, labels)):
        _, quads = textboxes_utils.read_sample(
            image_path=image_path,
            label_path=label_path
        )

        if len(quads) == 0:
            continue

        samples.append(f"{image_path} {label_path}")

    return samples
