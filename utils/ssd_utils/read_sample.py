import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from utils import pascal_voc_utils


def read_sample(image_path, label_path):
    """ Read image and label file in xml format.

    Args:
        - image_path: path to image file
        - label_path: path to label xml file

    Returns:
        - image: a numpy array with a data type of float
        - bboxes: a numpy array with a data type of float
        - classes: a list of strings

    Raises:
        - Image file does not exist
        - Label file does not exist
    """
    image_path = image_path.strip("\n")
    label_path = label_path.strip("\n")
    assert os.path.exists(image_path), "Image file does not exist."
    bboxes, classes = pascal_voc_utils.read_label(label_path)
    image = cv2.imread(image_path)  # read image in bgr format
    return np.array(image, dtype=np.float), np.array(bboxes, dtype=np.float), classes
