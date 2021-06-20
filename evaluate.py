import os
import json
import argparse
from glob import glob
from utils import data_utils

parser = argparse.ArgumentParser(
    description='Start the evaluation process of a particular network.')
parser.add_argument('images_dir', type=str, help='path to images dir.')
parser.add_argument('labels_dir', type=str, help='path to labels dir.')
parser.add_argument('predictions_dir', type=str, help='path to the predictions dir.')
parser.add_argument('split_file', type=str, help='path to the split file.')
parser.add_argument('--label_maps', type=str, help='path to label maps file.')
parser.add_argument('--output_dir', type=str,
                    help='path to output file.', default="output")
parser.add_argument("--iou_threshold", type=float, help="iou between gt box and pred box to be counted as a positive.", default=0.5)
args = parser.parse_args()

assert os.path.exists(args.images_dir), "images_dir does not exist"
assert os.path.exists(args.labels_dir), "labels_dir does not exist"
assert os.path.exists(args.predictions_dir), "labels_dir does not exist"
assert os.path.exists(args.split_file), "split_file does not exist"
assert args.iou_threshold > 0, "iou_threshold must be larger than zeros"

if args.label_maps is not None:
    assert os.path.exists(args.label_maps), "label_maps file does not exist"

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

prediction_files = glob(os.path.join(args.predictions_dir, "*txt"))

predictions_dict = {}
for prediction_file in prediction_files:
    filename = os.path.basename(prediction_file)
    filename = filename[:filename.index(".")]
    with open(prediction_file, "r") as pred_file:
        predictions = pred_file.readlines()
        predictions = [p.strip("\n").split(" ") for p in predictions]
        predictions = [{
            "class": p[0],
            "confidence_score": float(p[1]),
            "xmin": int(p[2]),
            "ymin": int(p[3]),
            "xmax": int(p[4]),
            "ymax": int(p[5])
        } for p in predictions]
        predictions_dict[filename] = predictions

samples = data_utils.get_samples_from_split(args.split_file, args.images_dir, args.labels_dir)
print(samples)

# print(prediction_files)
