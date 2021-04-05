import os
import cv2
from PIL import Image
import argparse
from glob import glob
import json
import numpy as np
import shutil
import os
import csv
import random
import xml.etree.ElementTree as ET

import numpy as np


def get_bboxes_from_quads(quads):
    xmin = np.min(quads[:, 0], axis=-1, keepdims=True)
    ymin = np.min(quads[:, 1], axis=-1, keepdims=True)
    xmax = np.max(quads[:, 0], axis=-1, keepdims=True)
    ymax = np.max(quads[:, 1], axis=-1, keepdims=True)
    return np.concatenate([xmin, ymin, xmax, ymax], axis=-1)


parser = argparse.ArgumentParser(
    description='Converts the MIDV500 dataset to a format suitable for training qssd with this repo.')
parser.add_argument('dataset_dir', type=str, help='path to dataset dir.')
parser.add_argument('output_dir', type=str, help='path to output dir.')
args = parser.parse_args()

assert os.path.exists(args.dataset_dir), "dataset_dir does not exist"
out_images_dir = os.path.join(args.output_dir, "images")
out_labels_dir = os.path.join(args.output_dir, "labels")
os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_labels_dir, exist_ok=True)

images = sorted(
    list(glob(os.path.join(args.dataset_dir, "*/images/*/*tif"))))
labels = sorted(
    list(glob(os.path.join(args.dataset_dir, "*/ground_truth/*/*json"))))

label_maps = list(glob(os.path.join(args.dataset_dir, "*")))
label_maps = sorted([i.split("/")[-1] for i in label_maps])

data_file = open(os.path.join(args.output_dir, "data.csv"), "w")
data_file_writer = csv.writer(
    data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
data_file_writer.writerow([
    "filename",
    "image_width",
    "image_height",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "poly_x1",
    "poly_y1",
    "poly_x2",
    "poly_y2",
    "poly_x3",
    "poly_y3",
    "poly_x4",
    "poly_y4",
    "class"
])

samples = []

for i, (image_file, label_file) in enumerate(list(zip(images, labels))):
    print(f"{i+1}/{len(images)}")
    filename = image_file.split("/")[-1]
    filename = filename[:filename.index(".")]
    label = image_file.split("/")[-4]

    image = Image.open(image_file)
    image.save(os.path.join(out_images_dir, f"{filename}.jpg"))
    with open(label_file, "r") as label_json_file:
        polygon = np.array(json.load(label_json_file)["quad"])
        bbox = get_bboxes_from_quads(polygon)
        xml_root = ET.Element("annotation")
        xml_filename = ET.SubElement(
            xml_root, "filename").text = f"{filename}.jpg"
        xml_size = ET.SubElement(xml_root, "size")
        xml_size_width = ET.SubElement(
            xml_size, "width").text = str(image.width)
        xml_size_height = ET.SubElement(
            xml_size, "height").text = str(image.height)
        xml_size_depth = ET.SubElement(
            xml_size, "depth").text = str(3)

        xml_object = ET.SubElement(xml_root, "object")
        xml_object_name = ET.SubElement(
            xml_object, "name").text = label
        xml_object_bndbox = ET.SubElement(xml_object, "bndbox")
        xml_object_polygon = ET.SubElement(
            xml_object, "polygon")
        xml_object_bndbox_xmin = ET.SubElement(
            xml_object_bndbox, "xmin").text = str(bbox[0])
        xml_object_bndbox_ymin = ET.SubElement(
            xml_object_bndbox, "ymin").text = str(bbox[1])
        xml_object_bndbox_xmax = ET.SubElement(
            xml_object_bndbox, "xmax").text = str(bbox[2])
        xml_object_bndbox_ymax = ET.SubElement(
            xml_object_bndbox, "ymax").text = str(bbox[3])
        xml_object_polygon_x1 = ET.SubElement(
            xml_object_polygon, "x1").text = str(polygon[0, 0])
        xml_object_polygon_y1 = ET.SubElement(
            xml_object_polygon, "y1").text = str(polygon[0, 1])
        xml_object_polygon_x2 = ET.SubElement(
            xml_object_polygon, "x2").text = str(polygon[1, 0])
        xml_object_polygon_y2 = ET.SubElement(
            xml_object_polygon, "y2").text = str(polygon[1, 1])
        xml_object_polygon_x3 = ET.SubElement(
            xml_object_polygon, "x3").text = str(polygon[2, 0])
        xml_object_polygon_y3 = ET.SubElement(
            xml_object_polygon, "y3").text = str(polygon[2, 1])
        xml_object_polygon_x4 = ET.SubElement(
            xml_object_polygon, "x4").text = str(polygon[3, 0])
        xml_object_polygon_y4 = ET.SubElement(
            xml_object_polygon, "y4").text = str(polygon[3, 1])
        data_file_writer.writerow([
            f"{filename}.xml",
            image.width,
            image.height,
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[2]),
            int(bbox[3]),
            int(polygon[0, 0]),
            int(polygon[0, 1]),
            int(polygon[1, 0]),
            int(polygon[1, 1]),
            int(polygon[2, 0]),
            int(polygon[2, 1]),
            int(polygon[3, 0]),
            int(polygon[3, 1]),
            label
        ])
        xml_tree = ET.ElementTree(xml_root)
        with open(os.path.join(out_labels_dir, f"{filename}.xml"), "wb+") as xml_file:
            xml_tree.write(xml_file)

        sample = f"{filename}.jpg {filename}.xml"
        samples.append(sample)

random.shuffle(samples)

train_idx = int(len(samples) * 0.8)
train_samples = samples[:train_idx]
val_test_samples = samples[train_idx:]
val_idx = int(len(val_test_samples) * 0.5)
val_samples = val_test_samples[:val_idx]
test_samples = val_test_samples[val_idx:]


def save_sample_to_split_file(s, f):
    split_file = open(os.path.join(args.output_dir, f), "w")
    for sample in s:
        split_file.write(f"{sample}\n")
    split_file.close()


print("-- saving training split")
save_sample_to_split_file(train_samples, "train.txt")
print("-- saving val split")
save_sample_to_split_file(val_samples, "val.txt")
print("-- saving test split")
save_sample_to_split_file(test_samples, "test.txt")
print("-- saving label_maps")
save_sample_to_split_file(label_maps, "label_maps.txt")
