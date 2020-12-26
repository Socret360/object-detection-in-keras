import cv2
import json
import argparse
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from networks import SSD300_VGG16
from utils import ssd_utils

parser = argparse.ArgumentParser(description='Start the training process of a particular network.')
parser.add_argument('input_image', type=str, help='path to the input image.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('label_maps', type=str, help='path to label maps file.')
parser.add_argument('weights', type=str, help='path to the weight file.')
parser.add_argument('--threshold', type=float, help='score.')
args = parser.parse_args()

with open(args.label_maps, "r") as file:
    label_maps = [line.strip("\n") for line in file.readlines()]

with open(args.config, "r") as config_file:
    config = json.load(config_file)

input_size = config["model"]["input_size"]
model = SSD300_VGG16(
    config,
    label_maps,
    is_training=False
)
label_maps = ["__backgroud__"] + label_maps
num_classes = len(label_maps)
model.load_weights(args.weights)

image = cv2.imread(args.input_image)
image_height, image_width, _ = image.shape
height_scale, width_scale = input_size/image_height, input_size/image_width
image = cv2.resize(image, (input_size, input_size))
t_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = preprocess_input(image)
image = np.expand_dims(image, axis=0)
y_pred = model.predict(image)

for i in y_pred[0]:
    print(f"--{label_maps[int(i[0])]}: {i[1]}")
    if i[1] < 1 and i[1] > args.threshold:
        cv2.putText(
            t_image,
            label_maps[int(i[0])],
            (int(i[2]), int(i[3])),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 0, 0),
            1,
            1
        )
        cv2.rectangle(
            t_image,
            (int(i[2]), int(i[3])),
            (int(i[4]), int(i[5])),
            (0, 255, 0),
            2
        )

cv2.imshow("image", t_image)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
