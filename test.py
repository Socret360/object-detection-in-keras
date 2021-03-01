import json
import cv2
import numpy as np
from glob import glob
from tensorflow.keras.applications import vgg16
from networks import TBPP_VGG16
from utils import textboxes_utils, augmentation_utils, ssd_utils, bbox_utils

with open("configs/tbpp384_vgg16.json", "r") as config_file:
    config = json.load(config_file)

print(config)
model = TBPP_VGG16(config)
model.summary()
