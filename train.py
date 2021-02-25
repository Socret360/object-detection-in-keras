import os
import json
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from callbacks import ModelCheckpoint
from tensorflow.keras.applications import vgg16, mobilenet, mobilenet_v2
import distutils
from losses import SSD_LOSS, TBPP_LOSS
from utils import data_utils, training_utils, command_line_utils
from networks import SSD_VGG16, SSD_MOBILENET, SSD_MOBILENETV2, TBPP_VGG16
from data_generators import SSD_DATA_GENERATOR, TBPP_DATA_GENERATOR

parser = argparse.ArgumentParser(description='Start the training process of a particular network.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('images_dir', type=str, help='path to images dir.')
parser.add_argument('labels_dir', type=str, help='path to labels dir.')
parser.add_argument('--training_split', type=str, help='path to training split file.')
parser.add_argument('--label_maps', type=str, help='path to label maps file.')
parser.add_argument('--learning_rate', type=float, help='learning rate used in training.', default=10e-3)
parser.add_argument('--epochs', type=int, help='the number of epochs to train', default=100)
parser.add_argument('--batch_size', type=int, help='the batch size used in training', default=32)
parser.add_argument('--shuffle', type=command_line_utils.str2bool, nargs='?', help='whether to shuffle the dataset when creating the batch', default=True)
parser.add_argument('--augment', type=command_line_utils.str2bool, nargs='?', help='whether to augment training samples', default=True)
parser.add_argument('--output_dir', type=str, help='path to config file.', default="output")
parser.add_argument('--checkpoint_weights', type=str, help='path to checkpoint weight file.')
parser.add_argument('--checkpoint_type', type=str, help='the type of checkpoint to save. One of: epoch or iteration, none', default="epoch")
parser.add_argument('--checkpoint_frequency', type=int, help='the frequency in which to save a model', default=1)
args = parser.parse_args()

assert os.path.exists(args.config), "config file does not exist"
assert os.path.exists(args.images_dir), "images_dir does not exist"
assert os.path.exists(args.labels_dir), "labels_dir does not exist"
assert args.epochs > 0, "epochs must be larger than zero"
assert args.batch_size > 0, "batch_size must be larger than 0"
assert args.learning_rate > 0, "learning_rate must be larger than 0"

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(args.config, "r") as config_file:
    config = json.load(config_file)

model_config = config["model"]
training_config = config["training"]

if model_config["name"] == "ssd_vgg16":
    model, generator, optimizer, loss, training_samples = training_utils.train_ssd_vgg16(config, args)
elif model_config["name"] == "ssd_mobilenetv1":
    model, generator, optimizer, loss, training_samples = training_utils.train_ssd_mobilenetv1(config, args)
elif model_config["name"] == "ssd_mobilenetv2":
    model, generator, optimizer, loss, training_samples = training_utils.train_ssd_mobilenetv2(config, args)
elif model_config["name"] == "tbpp_vgg16":
    model, generator, optimizer, loss, training_samples = training_utils.train_tbpp_vgg16(config, args)
else:
    print(f"model with name ${model_config['name']} has not been implemented yet")
    exit()


if args.checkpoint_weights is not None:
    assert os.path.exists(args.checkpoint_weights), "checkpoint_weights does not exist"
    model.load_weights(args.checkpoint_weights, by_name=True)

# model.summary()

num_iterations_per_epoch = len(training_samples)//args.batch_size

assert args.checkpoint_type in ["epoch", "iteration", "none"], "checkpoint_type must be one of epoch, iteration, none."

if args.checkpoint_type == "epoch":
    assert args.checkpoint_frequency < args.epochs, "checkpoint_frequency must be smaller than epochs."
elif args.checkpoint_type == "iteration":
    assert args.checkpoint_frequency < num_iterations_per_epoch * args.epochs, "checkpoint_frequency must be smaller than num_iterations_per_epoch * args.epochs"

model.fit(
    x=generator,
    batch_size=args.batch_size,
    epochs=args.epochs,
    steps_per_epoch=num_iterations_per_epoch,
    callbacks=[
        ModelCheckpoint(
            output_dir=args.output_dir,
            epoch_frequency=args.checkpoint_frequency if args.checkpoint_type == "epoch" else None,
            iteration_frequency=args.checkpoint_frequency if args.checkpoint_type == "iteration" else None,
        )
    ]
)
