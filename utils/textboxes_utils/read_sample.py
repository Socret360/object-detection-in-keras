import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np


def read_sample(image_path, label_path):
    """ Read image and label file in xml format.

    Args:
        - image_path: path to image file
        - label_path: path to label xml file

    Returns:
        - image: a numpy array with a data type of float
        - quads: a numpy array with a data type of float

    Raises:
        - Image file does not exist
        - Label file does not exist
    """
    image_path = image_path.strip("\n")
    label_path = label_path.strip("\n")
    assert os.path.exists(image_path), "Image file does not exist."
    assert os.path.exists(label_path), "Label file does not exist."

    image = cv2.imread(image_path)  # read image in bgr format

    with open(label_path, "r") as label_file:
        labels = label_file.readlines()
        labels = [label.strip("\ufeff").strip("\n") for label in labels]
        labels = [[int(i) for i in label.split(",")[:-1]] for label in labels]
        labels = np.array(labels)
        quads = np.reshape(labels, (labels.shape[0], 4, 2))

    return np.array(image, dtype=np.float), np.array(quads, dtype=np.float)
