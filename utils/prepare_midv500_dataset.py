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
import argparse


def get_bboxes_from_quads(quads):
    xmin = np.min(quads[:, 0], axis=-1, keepdims=True)
    ymin = np.min(quads[:, 1], axis=-1, keepdims=True)
    xmax = np.max(quads[:, 0], axis=-1, keepdims=True)
    ymax = np.max(quads[:, 1], axis=-1, keepdims=True)
    return np.concatenate([xmin, ymin, xmax, ymax], axis=-1)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(
    description='Converts the MIDV500 dataset to a format suitable for training qssd with this repo.')
parser.add_argument('dataset_dir', type=str, help='path to dataset dir.')
parser.add_argument('output_dir', type=str, help='path to output dir.')
parser.add_argument('--allow_card_over_edge', type=str2bool, nargs='?',
                    help='whether to allow the card to go over the edge of the image', default=True)
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
label_maps = sorted([os.path.basename(i) for i in label_maps])

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

t = list(zip(images, labels))

invalid = 0

for i, (image_file, label_file) in enumerate(t):
    print(f"{i+1}/{len(t)}")
    filename = os.path.basename(image_file)
    filename = filename[:filename.index(".")]

    label = image_file.split("/")[-4]

    with open(label_file, "r") as label_json_file:
        polygon = np.array(json.load(label_json_file)["quad"])
        image = cv2.imread(image_file)

        if not args.allow_card_over_edge:
            if np.any(np.reshape(polygon, (8)) < 0):
                print("-- invalid polygon: points go over left and top edge, skipped")
                invalid += 1
                continue
            elif np.any(np.reshape(polygon, (8))[[0, 2, 4, 6]] > image.width):
                print("-- invalid polygon: points go over right edge")
                invalid += 1
                continue
            elif np.any(np.reshape(polygon, (8))[[1, 3, 5, 7]] > image.height):
                print("-- invalid polygon: points go over bottom edge")
                invalid += 1
                continue

        cv2.imwrite(os.path.join(out_images_dir, f"{filename}.jpg"), image)

        bbox = get_bboxes_from_quads(polygon)
        xml_root = ET.Element("annotation")
        xml_filename = ET.SubElement(
            xml_root, "filename").text = f"{filename}.jpg"
        xml_size = ET.SubElement(xml_root, "size")
        xml_size_width = ET.SubElement(
            xml_size, "width").text = str(image.shape[1])
        xml_size_height = ET.SubElement(
            xml_size, "height").text = str(image.shape[0])
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
            image.shape[1],
            image.shape[0],
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

print(f"invalid: {invalid}, total: {len(t)}")
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
