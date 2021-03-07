import os
import cv2
import json
import argparse
import numpy as np
from glob import glob
from xml.dom import minidom
import xml.etree.cElementTree as ET
from pycocotools.coco import COCO
from textboxes_utils import read_sample

parser = argparse.ArgumentParser(description='Displays a sample')
parser.add_argument('image', type=str, help='path to image file.')
parser.add_argument('label', type=str, help='path to label file.')
args = parser.parse_args()

print("loading image file")

image, quads = read_sample(args.image, args.label)
image = np.uint8(image)

for quad in quads:
    cv2.polylines(
        image,
        [np.reshape(np.array(quad, dtype=np.int), (-1, 2))],
        True,
        (0, 255, 0),
        1
    )

cv2.imshow("image", image)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
