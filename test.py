import cv2
import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras.applications import vgg16, mobilenet, mobilenet_v2
import numpy as np
from glob import glob
from networks import SSD_VGG16, SSD_MOBILENET, SSD_MOBILENETV2
from utils import inference_utils, textboxes_utils, command_line_utils


parser = argparse.ArgumentParser(
    description='run inference on an input image.')
parser.add_argument('test_file', type=str, help='path to the test set file.')
parser.add_argument('images_dir', type=str, help='path to images dir.')
parser.add_argument('labels_dir', type=str, help='path to labels dir.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('weights', type=str, help='path to config file.')
parser.add_argument('--label_maps', type=str, help='path to label maps file.')
parser.add_argument('--num_predictions', type=int,
                    help='the number of detections to be output as final detections', default=10)
parser.add_argument('--output_dir', type=str,
                    help='ouput', default="output")
args = parser.parse_args()

assert os.path.exists(args.config), "config file does not exist"
assert args.num_predictions > 0, "num_predictions must be larger than zero"
# assert args.confidence_threshold > 0, "confidence_threshold must be larger than zero."
# assert args.confidence_threshold <= 1, "confidence_threshold must be smaller than or equal to 1."

with open(args.config, "r") as config_file:
    config = json.load(config_file)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

input_size = config["model"]["input_size"]
model_config = config["model"]

if model_config["name"] == "ssd_mobilenetv2":
    model, process_input_fn, label_maps = inference_utils.inference_ssd_mobilenetv2(
        config, args)
elif model_config["name"] == "ssd_vgg16":
    model, process_input_fn, label_maps = inference_utils.ssd_vgg16(config, args)
else:
    print(
        f"model with name ${model_config['name']} has not been implemented yet")
    exit()

model.load_weights(args.weights)

with open(args.test_file, "r") as test_set_file:
    tests = test_set_file.readlines()
    for idx, sample in enumerate(tests):
        print(f"{idx+1}/{len(tests)}")
        image_file, label_file = sample.split(" ")
        filename = image_file[:image_file.index(".")]
        image = cv2.imread(os.path.join(args.images_dir, image_file))
        image = np.array(image, dtype=np.float)
        image = np.uint8(image)
        image_height, image_width, _ = image.shape
        height_scale, width_scale = input_size/image_height, input_size/image_width

        image = cv2.resize(image, (input_size, input_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = process_input_fn(image)

        image = np.expand_dims(image, axis=0)
        y_pred = model.predict(image)[0]

        with open(os.path.join(args.output_dir, f"{filename}.txt"), "w") as outfile:
            for i, pred in enumerate(y_pred):
                classname = label_maps[int(pred[0]) - 1].lower()
                confidence_score = pred[1]
                pred[[2, 4]] /= width_scale
                pred[[3, 5]] /= height_scale
                outfile.write(f"{classname} {confidence_score} {int(pred[2])} {int(pred[3])} {int(pred[4])} {int(pred[5])}\n")
