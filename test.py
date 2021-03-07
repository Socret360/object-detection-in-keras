import json
import cv2
import numpy as np
from glob import glob
from tensorflow.keras.applications import vgg16
from networks import TBPP_VGG16
from utils import textboxes_utils, augmentation_utils, ssd_utils, bbox_utils

with open("/Users/socretlee/Downloads/synthtext-1/samples.txt", "r") as sample_file:
    lines = sample_file.readlines()
    print(len(lines))
