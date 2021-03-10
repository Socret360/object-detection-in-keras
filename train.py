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
from utils import data_utils, training_utils, command_line_utils, textboxes_utils
from networks import SSD_VGG16, SSD_MOBILENET, SSD_MOBILENETV2, TBPP_VGG16
from data_generators import SSD_DATA_GENERATOR, TBPP_DATA_GENERATOR

parser = argparse.ArgumentParser(description='Start the training process of a particular network.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('images_dir', type=str, help='path to images dir.')
parser.add_argument('labels_dir', type=str, help='path to labels dir.')
#
parser.add_argument('--training_split', type=str, help='path to training split file.')
parser.add_argument('--validation_split', type=str, help='path to validation split file.')
#
parser.add_argument('--label_maps', type=str, help='path to label maps file.')
#
parser.add_argument('--checkpoint', type=str, help='path to checkpoint weight file.')
parser.add_argument('--checkpoint_type', type=str, help='the type of checkpoint to save. One of: epoch or iteration, none', default="epoch")
parser.add_argument('--checkpoint_frequency', type=int, help='the frequency in which to save a model', default=1)
#
parser.add_argument('--learning_rate', type=float, help='learning rate used in training.', default=10e-3)
parser.add_argument('--epochs', type=int, help='the number of epochs to train', default=100)
parser.add_argument('--batch_size', type=int, help='the batch size used in training', default=32)
parser.add_argument('--shuffle', type=command_line_utils.str2bool, nargs='?', help='whether to shuffle the dataset when creating the batch', default=True)
parser.add_argument('--augment', type=command_line_utils.str2bool, nargs='?', help='whether to augment training samples', default=True)
parser.add_argument('--output_dir', type=str, help='path to config file.', default="output")
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

training_data_generator, num_training_samples, validation_data_generator, num_validation_samples = training_utils.get_data_generator(config, args)
model = training_utils.get_model(config, args)
loss = training_utils.get_loss(config, args)
optimizer = training_utils.get_optimizer(config, args)
model.compile(optimizer=optimizer, loss=loss.compute)

assert args.checkpoint_type in ["epoch", "iteration", "none"], "checkpoint_type must be one of epoch, iteration, none."
num_iterations_per_epoch = num_training_samples//args.batch_size
if args.checkpoint_type == "epoch":
    assert args.checkpoint_frequency < args.epochs, "checkpoint_frequency must be smaller than epochs."
elif args.checkpoint_type == "iteration":
    assert args.checkpoint_frequency < num_iterations_per_epoch * args.epochs, "checkpoint_frequency must be smaller than num_iterations_per_epoch * args.epochs"

if args.checkpoint is not None:
    assert os.path.exists(args.checkpoint), "checkpoint does not exist"
    model.load_weights(args.checkpoint, by_name=True)

model.fit(
    x=training_data_generator,
    validation_data=validation_data_generator,
    batch_size=args.batch_size,
    validation_batch_size=args.batch_size,
    epochs=args.epochs,
    callbacks=[
        ModelCheckpoint(
            output_dir=args.output_dir,
            epoch_frequency=args.checkpoint_frequency if args.checkpoint_type == "epoch" else None,
            iteration_frequency=args.checkpoint_frequency if args.checkpoint_type == "iteration" else None,
        )
    ]
)
model.save_weights(os.path.join(args.output_dir, "model.h5"))
