import os
import json
import argparse
from utils import evaluation_utils

parser = argparse.ArgumentParser(
    description='Start the evaluation process of a particular network.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('images_dir', type=str, help='path to images dir.')
parser.add_argument('labels_dir', type=str, help='path to labels dir.')
parser.add_argument('weights', type=str, help='path to the weight file.')
parser.add_argument('--label_maps', type=str, help='path to label maps file.')
parser.add_argument('--confidence_threshold', type=float,
                    help='the confidence score a detection should match in order to be counted.', default=0.9)
parser.add_argument('--num_predictions', type=int,
                    help='the number of detections to be output as final detections', default=10)
args = parser.parse_args()

assert os.path.exists(args.config), "config file does not exist"
assert os.path.exists(args.images_dir), "images_dir does not exist"
assert os.path.exists(args.labels_dir), "labels_dir does not exist"
assert os.path.exists(args.weights), "weights file does not exist"
assert args.num_predictions > 0, "num_predictions must be larger than zero"
assert args.confidence_threshold > 0, "confidence_threshold must be larger than zero."
assert args.confidence_threshold <= 1, "confidence_threshold must be smaller than or equal to 1."

with open(args.config, "r") as config_file:
    config = json.load(config_file)

model_config = config["model"]

if model_config["name"] == "qssd_vgg16":
    evaluation_utils.evaluate_qssd_vgg16(config, args)
elif model_config["name"] == "qssd_mobilenetv2":
    evaluation_utils.evaluate_qssd_mobilenetv2(config, args)
else:
    print(
        f"model with name ${model_config['name']} has not been implemented yet")
    exit()
