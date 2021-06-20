import os
import xml.etree.ElementTree as ET


def read_label(label_path):
    assert os.path.exists(label_path), "Label file does not exist."

    xml_root = ET.parse(label_path).getroot()
    objects = xml_root.findall("object")
    bboxes, classes = [], []
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

    return bboxes, classes
