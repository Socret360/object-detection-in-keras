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


parser = argparse.ArgumentParser(description='Converts the icdar 2013 dataset to a format suitable for training tbpp with this repo.')
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

print("-- copy images for training sets")
training_images = sorted(list(glob(os.path.join(args.dataset_dir, os.path.join("Challenge2_Training_Task12_Images", "*.jpg")))))
test_images = sorted(list(glob(os.path.join(args.dataset_dir, os.path.join("Challenge2_Test_Task12_Images", "*.jpg")))))

with open(os.path.join(args.output_dir, "train.txt"), "w") as train_split:
    for i, train_image in enumerate(training_images):
        print(f"image {i+1}/{len(training_images)}")
        image_filename = os.path.basename(train_image)
        label_filename = f"gt_{image_filename[:image_filename.index('.')]}.txt"
        shutil.copy(
            os.path.join(os.path.join(args.dataset_dir, "Challenge2_Training_Task12_Images"), image_filename),
            os.path.join(os.path.join(train_dir, "images"), image_filename)
        )
        with open(os.path.join(os.path.join(args.dataset_dir, "Challenge2_Training_Task1_GT"), label_filename), "r") as label_file:
            quads = label_file.readlines()
            with open(os.path.join(os.path.join(train_dir, "labels"), label_filename), "w") as output_label_file:
                for quad in quads:
                    quad = quad.strip("\n")
                    quad = quad.split(" ")
                    quad[-1] = quad[-1][1:-1]
                    quad = [i.strip(",") for i in quad]
                    quad[:4] = [float(i) for i in quad[:4]]
                    w = abs(quad[0] - quad[2])
                    h = abs(quad[1] - quad[3])
                    x1 = quad[0]
                    y1 = quad[1]
                    x2 = quad[0] + w
                    y2 = quad[1]
                    x3 = quad[0] + w
                    y3 = quad[1] + h
                    x4 = quad[0]
                    y4 = quad[1] + h
                    output_label_file.write(f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{quad[-1]}\n")
        train_split.write(f"{image_filename} {label_filename}\n")

with open(os.path.join(args.output_dir, "test.txt"), "w") as test_split:
    for i, test_image in enumerate(test_images):
        print(f"image {i+1}/{len(test_images)}")
        image_filename = os.path.basename(test_image)
        label_filename = f"gt_{image_filename[:image_filename.index('.')]}.txt"
        shutil.copy(
            os.path.join(os.path.join(args.dataset_dir, "Challenge2_Test_Task12_Images"), image_filename),
            os.path.join(os.path.join(testing_dir, "images"), image_filename)
        )
        with open(os.path.join(os.path.join(args.dataset_dir, "Challenge2_Test_Task1_GT"), label_filename), "r") as label_file:
            quads = label_file.readlines()
            with open(os.path.join(os.path.join(testing_dir, "labels"), label_filename), "w") as output_label_file:
                for quad in quads:
                    quad = quad.strip("\n")
                    quad = quad.split(" ")
                    quad[-1] = quad[-1][1:-1]
                    quad = [i.strip(",") for i in quad]
                    quad[:4] = [float(i) for i in quad[:4]]
                    w = abs(quad[0] - quad[2])
                    h = abs(quad[1] - quad[3])
                    x1 = quad[0]
                    y1 = quad[1]
                    x2 = quad[0] + w
                    y2 = quad[1]
                    x3 = quad[0] + w
                    y3 = quad[1] + h
                    x4 = quad[0]
                    y4 = quad[1] + h
                    output_label_file.write(f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{quad[-1]}\n")
        test_split.write(f"{image_filename} {label_filename}\n")
