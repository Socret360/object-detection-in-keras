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
    description='Converts the Pascal VOC 2007 and 2012 dataset to a format suitable for training tbpp with this repo.')
parser.add_argument('dataset_dir', type=str, help='path to dataset dir.')
parser.add_argument('output_dir', type=str, help='path to output dir.')
args = parser.parse_args()

assert os.path.exists(args.dataset_dir), "dataset_dir does not exist"
out_images_dir = os.path.join(args.output_dir, "images")
out_labels_dir = os.path.join(args.output_dir, "labels")
os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_labels_dir, exist_ok=True)


datasets = ["VOC2007", "VOC2012"]
train_samples, val_samples, trainval_samples, test_samples = [], [], [], []
for dataset in datasets:
    print(f"-- gather data from: {dataset}")
    dataset_dir = os.path.abspath(args.dataset_dir)
    dataset_dir = os.path.join(dataset_dir, dataset)
    images_dir = os.path.join(dataset_dir, "JPEGImages")
    labels_dir = os.path.join(dataset_dir, "Annotations")

    print(f"---- copy images")
    for image in list(glob(os.path.join(images_dir, "*jpg"))):
        destination_filename = os.path.basename(image)
        if dataset == "VOC2007":
            destination_filename = f"2007_{destination_filename}"
        dest = os.path.join(out_images_dir, destination_filename)
        shutil.copy(image, dest)

    print(f"---- copy labels")
    for label in list(glob(os.path.join(labels_dir, "*xml"))):
        destination_filename = os.path.basename(label)
        if dataset == "VOC2007":
            destination_filename = f"2007_{destination_filename}"
        dest = os.path.join(out_labels_dir, destination_filename)
        shutil.copy(label, dest)

    train_split = os.path.join(dataset_dir, "ImageSets/Main/train.txt")
    val_split = os.path.join(dataset_dir, "ImageSets/Main/val.txt")
    trainval_split = os.path.join(dataset_dir, "ImageSets/Main/trainval.txt")

    # train split
    print(f"---- gather train samples")
    with open(train_split, "r") as train_file:
        samples = train_file.readlines()
        for sample in samples:
            if dataset == "VOC2007":
                sample = "2007_" + sample.strip("\n")
            else:
                sample = sample.strip("\n")
            if sample not in train_samples:
                train_samples.append(sample)

    # val split
    print(f"---- gather val samples")
    with open(val_split, "r") as val_file:
        samples = val_file.readlines()
        for sample in samples:
            if dataset == "VOC2007":
                sample = "2007_" + sample.strip("\n")
            else:
                sample = sample.strip("\n")
            if sample not in val_samples:
                val_samples.append(sample)

    # trainval split
    print(f"---- gather trainval samples")
    with open(trainval_split, "r") as trainval_file:
        samples = trainval_file.readlines()
        for sample in samples:
            if dataset == "VOC2007":
                sample = "2007_" + sample.strip("\n")
            else:
                sample = sample.strip("\n")
            if sample not in trainval_samples:
                trainval_samples.append(sample)

    if dataset == "VOC2007":
        print(f"---- gather test samples")
        with open(os.path.join(dataset_dir, "ImageSets/Main/test.txt"), "r") as test_file:
            samples = test_file.readlines()
            for sample in samples:
                if dataset == "VOC2007":
                    sample = "2007_" + sample.strip("\n")
                else:
                    sample = sample.strip("\n")
                if sample not in test_samples:
                    test_samples.append(sample)


def save_samples_to_split(s, name):
    with open(os.path.join(args.output_dir, name), "w") as outfile:
        for i in s:
            outfile.write(f"{i}.jpg {i}.xml\n")


print(f"-- num_train: {len(train_samples)}")
save_samples_to_split(train_samples, "train.txt")
print(f"-- num_val: {len(val_samples)}")
save_samples_to_split(val_samples, "val.txt")
print(f"-- num_trainval: {len(trainval_samples)}")
save_samples_to_split(trainval_samples, "trainval.txt")
print(f"-- num_test: {len(test_samples)}")
save_samples_to_split(test_samples, "test.txt")

print(f"-- writing label_maps.txt")
dataset_dir = os.path.abspath(args.dataset_dir)
dataset_dir = os.path.join(dataset_dir, "VOC2007")

with open(os.path.join(args.output_dir, "label_maps.txt"), "w") as label_maps_file:
    labels = list(
        glob(os.path.join(dataset_dir, "ImageSets/Main/*_train.txt")))
    labels = [os.path.basename(i) for i in labels]
    labels = sorted([i[:i.index("_")] for i in labels])
    for classname in labels:
        label_maps_file.write(f"{classname}\n")
