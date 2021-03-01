import os
import cv2
import argparse
from glob import glob
from xml.dom import minidom
import xml.etree.cElementTree as ET
from pycocotools.coco import COCO

parser = argparse.ArgumentParser(description='Converts the coco dataset to a format suitable for training ssd with this repo.')
parser.add_argument('annotations_file', type=str, help='path to annotations file.')
parser.add_argument('images_dir', type=str, help='path to images dir.')
parser.add_argument('output_dir', type=str, help='path to output dir.')
args = parser.parse_args()

assert os.path.exists(args.annotations_file), "annotations_file does not exist"
assert os.path.exists(args.images_dir), "images_dir does not exist"
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
