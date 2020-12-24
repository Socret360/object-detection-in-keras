import os
import json
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from networks import SSD300_VGG16
from losses import SSD_LOSS
from data_generators import SSD_VOC_DATA_GENERATOR
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import voc_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the training process of a particular network.')
    parser.add_argument('config', type=str, help='path to config file.')
    parser.add_argument('label_maps', type=str, help='path to label maps file.')
    parser.add_argument('images_dir', type=str, help='path to images dir.')
    parser.add_argument('labels_dir', type=str, help='path to labels dir.')
    parser.add_argument('training_split', type=str, help='path to training split file.')
    parser.add_argument('learning_rate', type=float, help='learning rate used in training.', default=10e-3)
    parser.add_argument('--epochs', type=int, help='the number of epochs to train', default=100)
    parser.add_argument('--batch_size', type=int, help='the batch size used in training', default=32)
    parser.add_argument('--checkpoint_frequency', type=int, help='the number of epochs to save each checkpoint.', default=1)
    parser.add_argument('--shuffle', type=bool, help='whether to shuffle the dataset when creating the batch', default=True)
    parser.add_argument('--augment', type=bool, help='whether to augment training samples', default=True)
    parser.add_argument('--validiation_split', type=str, help='path to validiation split file.')
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
    training_samples = voc_utils.get_samples_from_split(
        split_file=args.training_split,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir
    )
    assert args.batch_size <= len(training_samples), "batch_size less than or equal to len(training_samples)"

    if args.validiation_split is not None:
        assert os.path.exists(args.validiation_split), "validiation_split file does not exist"
        validation_samples = voc_utils.get_samples_from_split(
            split_file=args.validiation_split,
            images_dir=args.images_dir,
            labels_dir=args.labels_dir
        )
        assert args.batch_size <= len(validation_samples), "batch_size less than or equal to len(validation_samples)"

    with open(args.label_maps, "r") as file:
        label_maps = [line.strip("\n") for line in file.readlines()]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.config, "r") as config_file:
        config = json.load(config_file)

    model_config = config["model"]
    training_config = config["training"]
    optimizer_config = training_config["optimizer"]

    if model_config["name"] == "ssd300_vgg16":
        model = SSD300_VGG16(config=config, label_maps=label_maps)
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
        model.compile(optimizer=optimizer, loss=loss.compute)
    else:
        print(f"model with name ${model_config['name']} has not been implemented yet")
        exit()

    if args.checkpoint_weights is not None and os.path.exists(args.checkpoint_weights):
        model.load_weights(args.checkpoint_weights, by_name=True)

    if validation_samples is not None:
        history = model.fit(
            x=SSD_VOC_DATA_GENERATOR(
                samples=training_samples,
                config=config,
                label_maps=label_maps,
                shuffle=args.shuffle,
                batch_size=args.batch_size,
                augment=args.augment,
            ),
            batch_size=args.batch_size,
            epochs=args.epochs,
            steps_per_epoch=len(training_samples)//args.batch_size,
            validation_data=SSD_VOC_DATA_GENERATOR(
                samples=validation_samples,
                config=config,
                label_maps=label_maps,
                shuffle=args.shuffle,
                batch_size=args.batch_size,
                augment=args.augment,
            ),
            validation_steps=len(validation_samples)//args.batch_size,
            callbacks=[
                ModelCheckpoint(
                    os.path.join(args.output_dir, 'cp_{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5'),
                    save_weights_only=True,
                    save_freq=(len(training_samples)//args.batch_size) * args.checkpoint_frequency
                ),
            ]
        )
        plt.legend(['train', 'val'], loc='upper left')
    else:
        history = model.fit(
            x=SSD_VOC_DATA_GENERATOR(
                samples=training_samples,
                config=config,
                label_maps=label_maps,
                shuffle=args.shuffle,
                batch_size=args.batch_size,
                augment=args.augment,
            ),
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
        plt.plot(history.history['val_loss'])
        plt.legend(['train'], loc='upper left')

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('output/training_graph.png')
