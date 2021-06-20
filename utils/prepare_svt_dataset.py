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
import os


parser = argparse.ArgumentParser(description='Converts the coco dataset to a format suitable for training ssd with this repo.')
parser.add_argument('dataset_dir', type=str, help='path to dataset dir.')
parser.add_argument('output_dir', type=str, help='path to output dir.')
args = parser.parse_args()

assert os.path.exists(args.dataset_dir), "dataset_dir does not exist"
testing_dir = os.path.join(os.path.join(args.output_dir, "test"))
train_dir = os.path.join(os.path.join(args.output_dir, "train"))
os.makedirs(os.path.join(testing_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(testing_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)

test = ET.parse(os.path.join(args.dataset_dir, "test.xml"))
train = ET.parse(os.path.join(args.dataset_dir, "train.xml"))

print("-- copy images for train sets")
training_images = train.getroot().findall("image")
for i, image in enumerate(training_images):
    print(f"image {i+1}/{len(training_images)}")
    image_filename = image.find("imageName").text
    image_filename = os.path.basename(image_filename)
    label_filename = f"{image_filename[:image_filename.index('.')]}.txt"
    rectangles = image.find("taggedRectangles").findall("taggedRectangle")

    shutil.copy(
        os.path.join(os.path.join(args.dataset_dir, "img"), image_filename),
        os.path.join(os.path.join(train_dir, "images"), image_filename)
    )

    with open(os.path.join(os.path.join(train_dir, "labels"), label_filename), "w") as label_file:
        for rectangle in rectangles:
            text = rectangle.find("tag").text
            xmin = int(rectangle.attrib["x"])
            ymin = int(rectangle.attrib["y"])
            width = int(rectangle.attrib["width"])
            height = int(rectangle.attrib["height"])
            label_file.write(f"{xmin},{ymin},{xmin+width},{ymin},{xmin+width},{ymin+height},{xmin},{ymin+height},{text}\n")

print("-- copy images for test sets")
testing_images = test.getroot().findall("image")
for i, image in enumerate(testing_images):
    print(f"image {i+1}/{len(testing_images)}")
    image_filename = image.find("imageName").text
    image_filename = os.path.basename(image_filename)
    label_filename = f"{image_filename[:image_filename.index('.')]}.txt"
    rectangles = image.find("taggedRectangles").findall("taggedRectangle")

    shutil.copy(
        os.path.join(os.path.join(args.dataset_dir, "img"), image_filename),
        os.path.join(os.path.join(testing_dir, "images"), image_filename)
    )

    with open(os.path.join(os.path.join(testing_dir, "labels"), label_filename), "w") as label_file:
        for rectangle in rectangles:
            text = rectangle.find("tag").text
            xmin = int(rectangle.attrib["x"])
            ymin = int(rectangle.attrib["y"])
            width = int(rectangle.attrib["width"])
            height = int(rectangle.attrib["height"])
            label_file.write(f"{xmin},{ymin},{xmin+width},{ymin},{xmin+width},{ymin+height},{xmin},{ymin+height},{text}\n")
