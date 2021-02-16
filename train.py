import os
import json
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import vgg16, mobilenet, mobilenet_v2

from losses import SSD_LOSS, TBPP_LOSS
from utils import data_utils
from networks import SSD_VGG16, SSD_MOBILENET, SSD_MOBILENETV2, TBPP_VGG16
from data_generators import SSD_DATA_GENERATOR, TBPP_DATA_GENERATOR

parser = argparse.ArgumentParser(description='Start the training process of a particular network.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('label_maps', type=str, help='path to label maps file.')
parser.add_argument('images_dir', type=str, help='path to images dir.')
parser.add_argument('labels_dir', type=str, help='path to labels dir.')
parser.add_argument('training_split', type=str, help='path to training split file.')
parser.add_argument('--learning_rate', type=float, help='learning rate used in training.', default=10e-3)
parser.add_argument('--epochs', type=int, help='the number of epochs to train', default=100)
parser.add_argument('--batch_size', type=int, help='the batch size used in training', default=32)
parser.add_argument('--checkpoint_frequency', type=int, help='the number of epochs to save each checkpoint.', default=1)
parser.add_argument('--shuffle', type=bool, help='whether to shuffle the dataset when creating the batch', default=True)
parser.add_argument('--augment', type=bool, help='whether to augment training samples', default=True)
parser.add_argument('--output_dir', type=str, help='path to config file.', default="output")
parser.add_argument('--checkpoint_weights', type=str, help='path to checkpoint weight file.')
args = parser.parse_args()

assert os.path.exists(args.training_split), "training_split file does not exist"
assert os.path.exists(args.label_maps), "label_maps file does not exist"
assert os.path.exists(args.config), "config file does not exist"
assert os.path.exists(args.images_dir), "images_dir does not exist"
assert os.path.exists(args.labels_dir), "labels_dir does not exist"
assert args.checkpoint_frequency > 0, "checkpoint_frequency must be larger than zero"
assert args.epochs > 0, "epochs must be larger than zero"
assert args.checkpoint_frequency <= args.epochs, "checkpoint_frequency must be less than or equals to epochs"
assert args.batch_size > 0, "batch_size must be larger than 0"
assert args.learning_rate > 0, "learning_rate must be larger than 0"

training_samples, validation_samples = None,  None
training_samples = data_utils.get_samples_from_split(
    split_file=args.training_split,
    images_dir=args.images_dir,
    labels_dir=args.labels_dir
)
assert args.batch_size <= len(training_samples), "batch_size less than or equal to len(training_samples)"

with open(args.label_maps, "r") as file:
    label_maps = [line.strip("\n") for line in file.readlines()]

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(args.config, "r") as config_file:
    config = json.load(config_file)

model_config = config["model"]
training_config = config["training"]

if model_config["name"] == "ssd_vgg16":
    process_input_fn = vgg16.preprocess_input
    model = SSD_VGG16(config=config, label_maps=label_maps)
    loss = SSD_LOSS(
        alpha=training_config["alpha"],
        min_negative_boxes=training_config["min_negative_boxes"],
        negative_boxes_ratio=training_config["negative_boxes_ratio"]
    )
    optimizer = SGD(
        lr=args.learning_rate,
        momentum=0.9,
        decay=0.0005,
        nesterov=False
    )
    generator = SSD_DATA_GENERATOR(
        samples=training_samples,
        config=config,
        label_maps=label_maps,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        augment=args.augment,
        process_input_fn=process_input_fn
    )
    model.compile(optimizer=optimizer, loss=loss.compute)
elif model_config["name"] == "ssd_mobilenetv1":
    process_input_fn = mobilenet.preprocess_input
    model = SSD_MOBILENET(config=config, label_maps=label_maps)
    loss = SSD_LOSS(
        alpha=training_config["alpha"],
        min_negative_boxes=training_config["min_negative_boxes"],
        negative_boxes_ratio=training_config["negative_boxes_ratio"]
    )
    optimizer = SGD(
        lr=args.learning_rate,
        momentum=0.9,
        decay=0.0005,
        nesterov=False
    )
    generator = SSD_DATA_GENERATOR(
        samples=training_samples,
        config=config,
        label_maps=label_maps,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        augment=args.augment,
        process_input_fn=process_input_fn
    )
    model.compile(optimizer=optimizer, loss=loss.compute)
elif model_config["name"] == "ssd_mobilenetv2":
    process_input_fn = mobilenet_v2.preprocess_input
    model = SSD_MOBILENETV2(config=config, label_maps=label_maps)
    loss = SSD_LOSS(
        alpha=training_config["alpha"],
        min_negative_boxes=training_config["min_negative_boxes"],
        negative_boxes_ratio=training_config["negative_boxes_ratio"]
    )
    optimizer = SGD(
        lr=args.learning_rate,
        momentum=0.9,
        decay=0.0005,
        nesterov=False
    )
    generator = SSD_DATA_GENERATOR(
        samples=training_samples,
        config=config,
        label_maps=label_maps,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        augment=args.augment,
        process_input_fn=process_input_fn
    )
    model.compile(optimizer=optimizer, loss=loss.compute)
elif model_config["name"] == "tbpp_vgg16":
    process_input_fn = vgg16.preprocess_input
    model = TBPP_VGG16(config=config, label_maps=label_maps)
    loss = TBPP_LOSS(
        alpha=training_config["alpha"],
        min_negative_boxes=training_config["min_negative_boxes"],
        negative_boxes_ratio=training_config["negative_boxes_ratio"]
    )
    optimizer = SGD(
        lr=args.learning_rate,
        momentum=0.9,
        decay=0.0005,
        nesterov=False
    )
    generator = TBPP_DATA_GENERATOR(
        samples=training_samples,
        config=config,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        augment=args.augment,
        process_input_fn=process_input_fn
    )
    model.compile(optimizer=optimizer, loss=loss.compute)
else:
    print(f"model with name ${model_config['name']} has not been implemented yet")
    exit()


if args.checkpoint_weights is not None:
    assert os.path.exists(args.checkpoint_weights), "checkpoint_weights does not exist"
    model.load_weights(args.checkpoint_weights, by_name=True)

history = model.fit(
    x=generator,
    batch_size=args.batch_size,
    epochs=args.epochs,
    steps_per_epoch=len(training_samples)//args.batch_size,
    callbacks=[
        ModelCheckpoint(
            os.path.join(args.output_dir, 'cp_{epoch:02d}_{loss:.4f}.h5'),
            save_weights_only=True,
            save_freq=(len(training_samples)//args.batch_size) * args.checkpoint_frequency
        ),
    ]
)

model.save_weights(os.path.join(args.output_dir, "model.h5"))
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig(os.path.join(args.output_dir, "training_graph.png"))
