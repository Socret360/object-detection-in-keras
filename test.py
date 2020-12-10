import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from models.ssd import SSD300_VGG16_ORIGINAL
from data_generators.voc import SSD_VOC_DATA_GENERATOR
from utils.ssd_utils import generate_default_boxes_for_feature_map, match_gt_boxes_to_default_boxes

with open("configs/ssd300_vgg16_original.json") as config_file:
    config = json.load(config_file)

data_generator = SSD_VOC_DATA_GENERATOR(
    samples=["data/test.jpg data/test.xml"],
    config=config
)

limit = 1

for i, (batch_x, batch_y) in enumerate(data_generator):
    print(f"batch {i+1}")
    for j in range(len(batch_x)):
        print(f"-- item {j}")
    if i >= limit:
        break
