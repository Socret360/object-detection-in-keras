import os
import json
import argparse
import tensorflow as tf
from networks import SSD_VGG16, SSD_MOBILENET, SSD_MOBILENETV2

SUPPORTED_TYPES = [
    "keras",
    "tflite"
]

parser = argparse.ArgumentParser(
    description='Converts a supported model into tflite.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('weights', type=str, help='path to the weight file.')
parser.add_argument('output_dir', type=str, help='path to the output folder.')
parser.add_argument('--label_maps', type=str, help='path to label maps file.')
parser.add_argument('--output_type', type=str,
                    help='the type of the output model. One of type: "keras", "tflite"', default="keras")
parser.add_argument('--num_predictions', type=int,
                    help='the number of detections to be output as final detections', default=10)
args = parser.parse_args()

assert os.path.exists(args.label_maps), "label_maps file does not exist"
assert os.path.exists(args.config), "config file does not exist"
assert args.num_predictions > 0, "num_predictions must be larger than zero"
assert args.output_type in SUPPORTED_TYPES, f"{args.output_type} is not supported yet. Please choose one of type {SUPPORTED_TYPES}"

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


with open(args.config, "r") as config_file:
    config = json.load(config_file)

model_config = config["model"]

if model_config["name"] == "ssd_vgg16":
    with open(args.label_maps, "r") as file:
        label_maps = [line.strip("\n") for line in file.readlines()]
    model = SSD_VGG16(
        config,
        label_maps,
        is_training=False,
        num_predictions=args.num_predictions)

elif model_config["name"] == "ssd_mobilenetv1":
    with open(args.label_maps, "r") as file:
        label_maps = [line.strip("\n") for line in file.readlines()]
    model = SSD_MOBILENET(
        config,
        label_maps,
        is_training=False,
        num_predictions=args.num_predictions)
elif model_config["name"] == "ssd_mobilenetv2":
    with open(args.label_maps, "r") as file:
        label_maps = [line.strip("\n") for line in file.readlines()]
    model = SSD_MOBILENETV2(
        config,
        label_maps,
        is_training=False,
        num_predictions=args.num_predictions)
else:
    print(
        f"model with name ${model_config['name']} has not been implemented yet")
    exit()

model.load_weights(args.weights)

config_file_name = os.path.basename(args.config)
config_file_name = config_file_name[:config_file_name.index(".")]
if args.output_type == "keras":
    model.save(os.path.join(args.output_dir, f"{config_file_name}.h5"))
elif args.output_type == "tflite":
    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
        tf.float16,
    ]
    tflite_model = tflite_converter.convert()
    open(os.path.join(args.output_dir, f"{config_file_name}.tflite"), 'wb').write(
        tflite_model)
