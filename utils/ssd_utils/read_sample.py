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
        - bboxes: a numpy array with a data type of float
        - classes: a list of strings

    Raises:
        - Image file does not exist
        - Label file does not exist
    """
    image_path = image_path.strip("\n")
    label_path = label_path.strip("\n")
    assert os.path.exists(image_path), "Image file does not exist."
    assert os.path.exists(label_path), "Label file does not exist."

    image = cv2.imread(image_path)  # read image in bgr format
    bboxes, classes = [], []
    xml_root = ET.parse(label_path).getroot()
    objects = xml_root.findall("object")
    for i, obj in enumerate(objects):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        # the reason why we use float() is because some value in bndbox are float
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        bboxes.append([xmin, ymin, xmax, ymax])
        classes.append(name)
    return np.array(image, dtype=np.float), np.array(bboxes, dtype=np.float), classes
