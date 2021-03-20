import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np


def read_sample(image_path, label_path):
    image_path = image_path.strip("\n")
    label_path = label_path.strip("\n")
    assert os.path.exists(image_path), "Image file does not exist."
    assert os.path.exists(label_path), "Label file does not exist."

    image = cv2.imread(image_path)  # read image in bgr format
    polygons, classes = [], []
    xml_root = ET.parse(label_path).getroot()
    objects = xml_root.findall("object")
    for i, obj in enumerate(objects):
        name = obj.find("name").text
        polygon = obj.find("polygon")
        # the reason why we use float() is because some value in bndbox are float
        x1 = float(polygon.find("x1").text)
        y1 = float(polygon.find("y1").text)
        x2 = float(polygon.find("x2").text)
        y2 = float(polygon.find("y2").text)
        x3 = float(polygon.find("x3").text)
        y3 = float(polygon.find("y3").text)
        x4 = float(polygon.find("x4").text)
        y4 = float(polygon.find("y4").text)
        polygons.append([x1, y2, x2, y2, x3, y3, x4, y4])
        classes.append(name)

    
    polygons = np.array(polygons)
    polygons = np.reshape(polygons, (polygons.shape[0], 4, 2))

    return np.array(image, dtype=np.float), np.array(polygons, dtype=np.float), classes
