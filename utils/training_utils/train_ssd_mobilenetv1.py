import os

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import mobilenet
from losses import SSD_LOSS
from utils import data_utils
from networks import SSD_MOBILENET
from data_generators import SSD_DATA_GENERATOR


def train_ssd_mobilenetv1(config, args):
    """"""
    assert args.label_maps is not None, "please specify a label maps file for this model"
    assert os.path.exists(args.label_maps), "label_maps file does not exist"
    with open(args.label_maps, "r") as file:
        label_maps = [line.strip("\n") for line in file.readlines()]

    training_samples = data_utils.get_samples_from_split(
        split_file=args.training_split,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir
    )
    assert args.batch_size <= len(training_samples), "batch_size less than or equal to len(training_samples)"

    training_config = config["training"]
    model = SSD_MOBILENET(config=config)
    loss = SSD_LOSS(
        alpha=training_config["alpha"],
        min_negative_boxes=training_config["min_negative_boxes"],
        negative_boxes_ratio=training_config["negative_boxes_ratio"]
    )
    optimizer = SGD(lr=args.learning_rate, momentum=0.9, decay=0.0005, nesterov=False)
    generator = SSD_DATA_GENERATOR(
        samples=training_samples,
        label_maps=label_maps,
        config=config,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        augment=args.augment,
        process_input_fn=mobilenet.preprocess_input
    )
    model.compile(optimizer=optimizer, loss=loss.compute)
    return model, generator, optimizer, loss, training_samples
