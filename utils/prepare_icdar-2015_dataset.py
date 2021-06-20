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


parser = argparse.ArgumentParser(description='Converts the icdar 2015 dataset to a format suitable for training tbpp with this repo.')
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
training_images = sorted(list(glob(os.path.join(args.dataset_dir, os.path.join("ch4_training_images", "*.jpg")))))
test_images = sorted(list(glob(os.path.join(args.dataset_dir, os.path.join("ch4_test_images", "*.jpg")))))

with open(os.path.join(args.output_dir, "train.txt"), "w") as train_split:
    for i, train_image in enumerate(training_images):
        print(f"image {i+1}/{len(training_images)}")
        image_filename = os.path.basename(train_image)
        label_filename = f"gt_{image_filename[:image_filename.index('.')]}.txt"
        shutil.copy(
            os.path.join(os.path.join(args.dataset_dir, "ch4_training_images"), image_filename),
            os.path.join(os.path.join(train_dir, "images"), image_filename)
        )
        shutil.copy(
            os.path.join(os.path.join(args.dataset_dir, "ch4_training_localization_transcription_gt"), label_filename),
            os.path.join(os.path.join(train_dir, "labels"), label_filename)
        )
        train_split.write(f"{image_filename} {label_filename}\n")

with open(os.path.join(args.output_dir, "test.txt"), "w") as test_split:
    for i, test_image in enumerate(test_images):
        print(f"image {i+1}/{len(test_images)}")
        image_filename = os.path.basename(test_image)
        label_filename = f"gt_{image_filename[:image_filename.index('.')]}.txt"
        shutil.copy(
            os.path.join(os.path.join(args.dataset_dir, "ch4_test_images"), image_filename),
            os.path.join(os.path.join(testing_dir, "images"), image_filename)
        )
        shutil.copy(
            os.path.join(os.path.join(args.dataset_dir, "Challenge4_Test_Task1_GT"), label_filename),
            os.path.join(os.path.join(testing_dir, "labels"), label_filename)
        )
        test_split.write(f"{image_filename} {label_filename}\n")
