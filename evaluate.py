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
parser.add_argument('--output_dir', type=str,
                    help='path to output file.', default="output")
parser.add_argument('--split_file', type=str, help='path to split_file file.')
parser.add_argument("--iou_threshold", type=float, help="iou between gt box and pred box to be counted as a positive.", default=0.5)
parser.add_argument('--num_predictions', type=int,
                    help='the number of detections to be output as final detections', default=10)
args = parser.parse_args()

assert os.path.exists(args.config), "config file does not exist"
assert os.path.exists(args.images_dir), "images_dir does not exist"
assert os.path.exists(args.labels_dir), "labels_dir does not exist"
assert os.path.exists(args.split_file), "split_file does not exist"
assert os.path.exists(args.weights), "weights file does not exist"
assert args.num_predictions > 0, "num_predictions must be larger than zero"
if args.label_maps is not None:
    assert os.path.exists(args.label_maps), "label_maps file does not exist"

with open(args.config, "r") as config_file:
    config = json.load(config_file)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

model_config = config["model"]

if model_config["name"] == "ssd_vgg16":
    evaluation_utils.ssd_vgg16(config, args)
elif model_config["name"] == "ssd_mobilenetv2":
    evaluation_utils.ssd_mobilenetv2(config, args)
else:
    print(
        f"model with name ${model_config['name']} has not been implemented yet")
    exit()
