import os
import cv2
import json
import argparse
import numpy as np
from glob import glob
from xml.dom import minidom
import xml.etree.cElementTree as ET
from pycocotools.coco import COCO
from utils import ssd_utils

parser = argparse.ArgumentParser(description='Displays default boxes in a selected image.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('image', type=str, help='path to image file.')
args = parser.parse_args()

print("loading config file")
with open(args.config, "r") as config_file:
    config = json.load(config_file)


model_config = config["model"]
default_boxes_config = model_config["default_boxes"]
input_size = model_config["input_size"]
extra_box_for_ar_1 = default_boxes_config["extra_box_for_ar_1"]
clip_boxes = default_boxes_config["clip_boxes"]

print("loading image file")
image = cv2.imread(args.image)
image = cv2.resize(image, (input_size, input_size))

print("generating default boxes")
scales = np.linspace(
    default_boxes_config["min_scale"],
    default_boxes_config["max_scale"],
    len(default_boxes_config["layers"])
)
mbox_conf_layers = []
mbox_loc_layers = []
mbox_default_boxes_layers = []
for i, layer in enumerate(default_boxes_config["layers"]):
    temp_image = image.copy()
    print(f"displaying default boxes for layer: {layer['name']}")
    layer_default_boxes = ssd_utils.generate_default_boxes_for_feature_map(
        feature_map_size=layer["size"],
        image_size=input_size,
        offset=layer["offset"],
        scale=scales[i],
        next_scale=scales[i+1] if i+1 <= len(default_boxes_config["layers"]) - 1 else 1,
        aspect_ratios=layer["aspect_ratios"],
        variances=default_boxes_config["variances"],
        extra_box_for_ar_1=extra_box_for_ar_1,
        clip_boxes=clip_boxes
    )

    grid_size = input_size / layer["size"]
    offset = layer["offset"]
    offset_x, offset_y = offset

    cx = np.linspace(offset_x * grid_size, input_size - (offset_x * grid_size), layer["size"])
    cy = np.linspace(offset_y * grid_size, input_size - (offset_y * grid_size), layer["size"])

    for n in range(len(cx)):
        for m in range(len(cy)):
            cv2.circle(
                temp_image,
                (int(cx[n]), int(cy[m])),
                1,
                (255, 0, 0),
                1
            )

    middle_cell = layer['size']//2
    target_cell = 0 if middle_cell == 0 else middle_cell

    for default_box in layer_default_boxes[target_cell][target_cell]:
        cx = default_box[0] * input_size
        cy = default_box[1] * input_size
        w = default_box[2] * input_size
        h = default_box[3] * input_size
        cv2.rectangle(
            temp_image,
            (int(cx-(w/2)), int(cy-(h/2))),
            (int(cx+(w/2)), int(cy+(h/2))),
            (0, 255, 0),
            3
        )
    cv2.imshow(f"layer: {layer['name']}", temp_image)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
