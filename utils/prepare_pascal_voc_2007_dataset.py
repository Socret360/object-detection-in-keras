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


parser = argparse.ArgumentParser(
    description='Converts the Pascal VOC 2007 dataset to a format suitable for training ssd with this repo.')
parser.add_argument('dataset_dir', type=str, help='path to dataset dir.')
parser.add_argument('output_dir', type=str, help='path to output dir.')
args = parser.parse_args()

assert os.path.exists(args.dataset_dir), "dataset_dir does not exist"
images_dir = os.path.join(args.dataset_dir, "JPEGImages")
labels_dir = os.path.join(args.dataset_dir, "Annotations")
out_images_dir = os.path.join(args.output_dir, "images")
out_labels_dir = os.path.join(args.output_dir, "labels")
os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_labels_dir, exist_ok=True)

print(f"-- creating split files")
print(f"---- train.txt")
with open(os.path.join(args.output_dir, "train.txt"), "w") as train_split_file:
    with open(os.path.join(args.dataset_dir, "ImageSets/Main/train.txt"), "r") as train_file:
        samples = train_file.readlines()
        for i, sample in enumerate(samples):
            sample = sample.strip("\n")
            train_split_file.write(f"{sample}.jpg {sample}.xml\n")

print(f"---- val.txt")
with open(os.path.join(args.output_dir, "val.txt"), "w") as val_split_file:
    with open(os.path.join(args.dataset_dir, "ImageSets/Main/val.txt"), "r") as val_file:
        samples = val_file.readlines()
        for sample in samples:
            sample = sample.strip("\n")
            val_split_file.write(f"{sample}.jpg {sample}.xml\n")

print(f"---- test.txt")
with open(os.path.join(args.output_dir, "test.txt"), "w") as val_split_file:
    with open(os.path.join(args.dataset_dir, "ImageSets/Main/test.txt"), "r") as val_file:
        samples = val_file.readlines()
        for sample in samples:
            sample = sample.strip("\n")
            val_split_file.write(f"{sample}.jpg {sample}.xml\n")

print(f"---- trainval.txt")
with open(os.path.join(args.output_dir, "split.txt"), "w") as trainval_split_file:
    with open(os.path.join(args.dataset_dir, "ImageSets/Main/trainval.txt"), "r") as trainval_file:
        samples = trainval_file.readlines()
        for sample in samples:
            sample = sample.strip("\n")
            trainval_split_file.write(f"{sample}.jpg {sample}.xml\n")

print(f"-- copying images")
for i, sample in enumerate(list(glob(os.path.join(images_dir, "*jpg")))):
    filename = os.path.basename(sample)
    shutil.copy(
        sample,
        os.path.join(out_images_dir, filename)
    )

print(f"-- copying labels")
for i, sample in enumerate(list(glob(os.path.join(labels_dir, "*xml")))):
    filename = os.path.basename(sample)
    shutil.copy(
        sample,
        os.path.join(out_labels_dir, filename)
    )

print(f"-- writing label_maps.txt")
with open(os.path.join(args.output_dir, "label_maps.txt"), "w") as label_maps_file:
    labels = list(
        glob(os.path.join(args.dataset_dir, "ImageSets/Main/*_train.txt")))
    labels = [os.path.basename(i) for i in labels]
    labels = sorted([i[:i.index("_")] for i in labels])
    for classname in labels:
        label_maps_file.write(f"{classname}\n")
