import os
import json
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import training_utils, command_line_utils

parser = argparse.ArgumentParser(
    description='Start the training process of a particular network.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('images_dir', type=str, help='path to images dir.')
parser.add_argument('labels_dir', type=str, help='path to labels dir.')
#
parser.add_argument('--training_split', type=str,
                    help='path to training split file.')
parser.add_argument('--validation_split', type=str,
                    help='path to validation split file.')
#
parser.add_argument('--label_maps', type=str, help='path to label maps file.')
#
parser.add_argument('--checkpoint', type=str,
                    help='path to checkpoint weight file.')
#
parser.add_argument('--learning_rate', type=float,
                    help='learning rate used in training.', default=10e-3)
parser.add_argument('--epochs', type=int,
                    help='the number of epochs to train', default=100)
parser.add_argument('--batch_size', type=int,
                    help='the batch size used in training', default=32)
parser.add_argument('--shuffle', type=command_line_utils.str2bool, nargs='?',
                    help='whether to shuffle the dataset when creating the batch', default=True)
parser.add_argument('--augment', type=command_line_utils.str2bool,
                    nargs='?', help='whether to augment training samples', default=False)
parser.add_argument('--output_dir', type=str,
                    help='path to config file.', default="output")
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

training_data_generator, num_training_samples, validation_data_generator, num_validation_samples = training_utils.get_data_generator(
    config, args)
model = training_utils.get_model(config, args)
loss = training_utils.get_loss(config, args)
optimizer = training_utils.get_optimizer(config, args)
model.compile(optimizer=optimizer, loss=loss.compute)

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
            filepath=os.path.join(
                args.output_dir, "cp_{epoch:02d}_loss-{loss:.2f}_valloss-{val_loss:.2f}.hdf5"),
            save_weights_only=True,
            monitor='val_loss',
            mode='max'
        )
    ]
)
model.save_weights(os.path.join(args.output_dir, "model.h5"))
