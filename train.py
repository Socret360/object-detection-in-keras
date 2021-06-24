from utils import training_utils, command_line_utils
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN, LearningRateScheduler
import argparse
import json
import os


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
parser.add_argument('--initial_epoch', type=int,
                    help='the initial epochs to start from', default=0)
parser.add_argument('--batch_size', type=int,
                    help='the batch size used in training', default=32)
parser.add_argument('--shuffle', type=command_line_utils.str2bool, nargs='?',
                    help='whether to shuffle the dataset when creating the batch', default=True)
parser.add_argument('--augment', type=command_line_utils.str2bool,
                    nargs='?', help='whether to augment training samples', default=False)
parser.add_argument('--schedule_lr', type=command_line_utils.str2bool,
                    nargs='?', help='whether to use the lr scheduler', default=True)
parser.add_argument('--show_network_structure', type=command_line_utils.str2bool,
                    nargs='?', help='whether to print out the network structure when constructing the network', default=False)
parser.add_argument('--output_dir', type=str,
                    help='path to config file.', default="output")
args = parser.parse_args()

assert os.path.exists(args.config), "config file does not exist"
assert os.path.exists(args.images_dir), "images_dir does not exist"
assert os.path.exists(args.labels_dir), "labels_dir does not exist"
assert args.epochs > 0, "epochs must be larger than zero"
assert args.batch_size > 0, "batch_size must be larger than 0"
assert args.learning_rate > 0, "learning_rate must be larger than 0"

if args.label_maps is not None:
    assert os.path.exists(args.label_maps), "label_maps file does not exist"

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(args.config, "r") as config_file:
    config = json.load(config_file)

model_config = config["model"]

if model_config["name"] == "ssd_mobilenetv1":
    training_utils.ssd_mobilenetv1(config, args)
elif model_config["name"] == "ssd_mobilenetv2":
    training_utils.ssd_mobilenetv2(config, args)
elif model_config["name"] == "ssd_vgg16":
    # configure callbacks here
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(
                args.output_dir,
                "cp_{epoch:02d}_loss-{loss:.2f}.h5" if args.validation_split is None else "cp_{epoch:02d}_loss-{loss:.2f}_valloss-{val_loss:.2f}.h5"
            ),
            save_weights_only=False,
            save_best_only=True,
            monitor='loss' if args.validation_split is None else 'val_loss',
            mode='min'
        ),
        CSVLogger(
            os.path.join(args.output_dir, "training.csv"),
            append=False
        ),
        TerminateOnNaN(),
    ]

    if (args.schedule_lr):
        def lr_schedule(epoch):
            if epoch < 108:
                return args.learning_rate
            elif epoch < 146:
                return 0.0001
            else:
                return 0.00001
        callbacks.append(LearningRateScheduler(schedule=lr_schedule, verbose=1))

    training_utils.ssd_vgg16(config, args, callbacks)
elif model_config["name"] == "tbpp_vgg16":
    training_utils.tbpp_vgg16(config, args)
else:
    print(
        f"model with name ${model_config['name']} has not been implemented yet")
    exit()
