import os
import cv2
import argparse
from glob import glob
import json
from xml.dom import minidom
import xml.etree.cElementTree as ET
from data_utils import COCO_Text
import numpy as np
import shutil

parser = argparse.ArgumentParser(description='Converts the coco dataset to a format suitable for training ssd with this repo.')
parser.add_argument('annotations_file', type=str, help='path to annotations file.')
parser.add_argument('images_dir', type=str, help='path to images dir.')
parser.add_argument('output_dir', type=str, help='path to output dir.')
args = parser.parse_args()

assert os.path.exists(args.annotations_file), "annotations_file does not exist"
assert os.path.exists(args.images_dir), "images_dir does not exist"
val_dir = os.path.join(os.path.join(args.output_dir, "val"))
train_dir = os.path.join(os.path.join(args.output_dir, "train"))
os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)

coco = COCO_Text(annotation_file=args.annotations_file)

print("-- copying images for validation sets")
for i, image_id in enumerate(coco.val):
    annotations = coco.loadAnns(coco.getAnnIds([image_id]))
    image_info = coco.loadImgs([image_id])[0]
    image_filename = image_info["file_name"]

    if len(annotations) == 0:
        print(f"\n---- skipped: {image_filename}\n")
        continue

    filter_annotations = []

    for annotation in annotations:
        quad = annotation["mask"]
        text = annotation["utf8_string"]
        if len(quad) != 8 or annotation["utf8_string"] == "":
            continue
        filter_annotations.append(annotation)

    if len(filter_annotations) == 0:
        print(f"\n---- skipped: {image_filename}\n")
        continue

    shutil.copy(
        os.path.join(args.images_dir, image_filename),
        os.path.join(os.path.join(val_dir, "images"), image_filename)
    )

    label_file_name = f"{image_filename[:image_filename.index('.')]}.txt"

    with open(os.path.join(os.path.join(val_dir, "labels"), label_file_name), "w") as label_file:
        for annotation in filter_annotations:
            quad = annotation["mask"]
            legibility = annotation["legibility"]
            text = "###" if legibility == "illegible" else annotation["utf8_string"]
            for num in quad:
                label_file.write(f"{int(num)},")
            label_file.write(f"{text}\n")

    # xml_root = ET.Element("annotation")
    # xml_filename = ET.SubElement(xml_root, "filename").text = image_filename
    # xml_size = ET.SubElement(xml_root, "size")
    # xml_size_width = ET.SubElement(xml_size, "width").text = str(image_info["width"])
    # xml_size_height = ET.SubElement(xml_size, "height").text = str(image_info["height"])
    # xml_size_depth = ET.SubElement(xml_size, "depth").text = str(3)
    # for annotation in annotations:
    #     category_id = annotation['category_id']
    #     bbox = annotation['bbox']
    #     label = coco.cats[category_id]["name"]
    #     xml_object = ET.SubElement(xml_root, "object")
    #     xml_object_name = ET.SubElement(xml_object, "name").text = label
    #     xml_object_bndbox = ET.SubElement(xml_object, "bndbox")
    #     xml_object_bndbox_xmin = ET.SubElement(xml_object_bndbox, "xmin").text = str(bbox[0])
    #     xml_object_bndbox_ymin = ET.SubElement(xml_object_bndbox, "ymin").text = str(bbox[1])
    #     xml_object_bndbox_xmax = ET.SubElement(xml_object_bndbox, "xmax").text = str(bbox[0] + bbox[2])
    #     xml_object_bndbox_ymax = ET.SubElement(xml_object_bndbox, "ymax").text = str(bbox[1] + bbox[3])
    # xml_tree = ET.ElementTree(xml_root)
    # xml_file_name = f"{image_filename[:image_filename.index('.')]}.xml"
    # with open(os.path.join(args.output_dir, xml_file_name), "wb+") as xml_file:
    #     xml_tree.write(xml_file)
    #     split_file.write(f"{image_filename} {xml_file_name}\n")
    #     num_samples += 1


# print(coco.)
