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
with open(os.path.join(args.output_dir, "test.txt"), "w") as test_file:
    for i, image_id in enumerate(coco.val):
        print(f"image {i+1} / {len(coco.val)}")
        annotations = coco.loadAnns(coco.getAnnIds([image_id]))
        image_info = coco.loadImgs([image_id])[0]
        image_filename = image_info["file_name"]

        if len(annotations) == 0:
            continue

        filter_annotations = []

        for annotation in annotations:
            quad = annotation["polygon"]
            if len(quad) != 8:
                continue
            filter_annotations.append(annotation)

        if len(filter_annotations) == 0:
            continue

        shutil.copy(
            os.path.join(args.images_dir, image_filename),
            os.path.join(os.path.join(val_dir, "images"), image_filename)
        )

        label_file_name = f"{image_filename[:image_filename.index('.')]}.txt"

        with open(os.path.join(os.path.join(val_dir, "labels"), label_file_name), "w") as label_file:
            for annotation in filter_annotations:
                quad = annotation["polygon"]
                try:
                    text = annotation["utf8_string"]
                except:
                    text = "###"
                for num in quad:
                    label_file.write(f"{float(num)},")
                label_file.write(f"{text}\n")

        test_file.write(f"{image_filename} {label_file_name}\n")

print("-- copying images for training sets")
with open(os.path.join(args.output_dir, "train.txt"), "w") as train_file:
    for i, image_id in enumerate(coco.train):
        print(f"image {i+1} / {len(coco.train)}")
        annotations = coco.loadAnns(coco.getAnnIds([image_id]))
        image_info = coco.loadImgs([image_id])[0]
        image_filename = image_info["file_name"]

        if len(annotations) == 0:
            continue

        filter_annotations = []

        for annotation in annotations:
            quad = annotation["polygon"]
            if len(quad) != 8:
                continue
            filter_annotations.append(annotation)

        if len(filter_annotations) == 0:
            continue

        shutil.copy(
            os.path.join(args.images_dir, image_filename),
            os.path.join(os.path.join(train_dir, "images"), image_filename)
        )

        label_file_name = f"{image_filename[:image_filename.index('.')]}.txt"

        with open(os.path.join(os.path.join(train_dir, "labels"), label_file_name), "w") as label_file:
            for annotation in filter_annotations:
                quad = annotation["polygon"]
                try:
                    text = annotation["utf8_string"]
                except:
                    text = "###"
                for num in quad:
                    label_file.write(f"{float(num)},")
                label_file.write(f"{text}\n")

        train_file.write(f"{image_filename} {label_file_name}\n")
