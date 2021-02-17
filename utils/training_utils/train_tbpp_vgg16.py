from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import vgg16
from losses import TBPP_LOSS
from utils import textboxes_utils
from networks import TBPP_VGG16
from data_generators import TBPP_DATA_GENERATOR


def train_tbpp_vgg16(config, args):
    """"""
    training_samples = textboxes_utils.get_samples(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir
    )
    assert args.batch_size <= len(training_samples), "batch_size less than or equal to len(training_samples)"

    training_config = config["training"]
    model = TBPP_VGG16(config=config)
    loss = TBPP_LOSS(
        alpha=training_config["alpha"],
        min_negative_boxes=training_config["min_negative_boxes"],
        negative_boxes_ratio=training_config["negative_boxes_ratio"]
    )
    optimizer = SGD(lr=args.learning_rate, momentum=0.9, decay=0.0005, nesterov=False)
    generator = TBPP_DATA_GENERATOR(
        samples=training_samples,
        config=config,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        augment=args.augment,
        process_input_fn=vgg16.preprocess_input
    )
    model.compile(optimizer=optimizer, loss=loss.compute)
    return model, generator, optimizer, loss, training_samples
