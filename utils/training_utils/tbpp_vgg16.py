import os
from utils import data_utils
from losses import TBPP_LOSS
from networks import TBPP_VGG16
from tensorflow.keras.optimizers import Adam
from data_generators import TBPP_DATA_GENERATOR
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import preprocess_input


def tbpp_vgg16(config, args):
    training_config = config["training"]

    training_samples = data_utils.get_samples_from_split(
        split_file=args.training_split,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir
    )

    if args.validation_split is not None:
        validation_samples = data_utils.get_samples_from_split(
            split_file=args.validation_split,
            images_dir=args.images_dir,
            labels_dir=args.labels_dir
        )

    print("creating data generator for tbpp_vgg16")
    training_data_generator = TBPP_DATA_GENERATOR(
        samples=training_samples,
        config=config,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        augment=args.augment,
        process_input_fn=preprocess_input
    )

    if args.validation_split is not None:
        print("-- validation split specified")
        validation_data_generator = TBPP_DATA_GENERATOR(
            samples=validation_samples,
            config=config,
            shuffle=args.shuffle,
            batch_size=args.batch_size,
            augment=False,
            process_input_fn=preprocess_input
        )

    loss = TBPP_LOSS(
        alpha=training_config["alpha"],
        min_negative_boxes=training_config["min_negative_boxes"],
        negative_boxes_ratio=training_config["negative_boxes_ratio"]
    )

    model = TBPP_VGG16(
        config=config,
        is_training=True
    )

    optimizer = Adam(
        lr=args.learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=0.001,
        decay=0.0
    )

    model.compile(
        optimizer=optimizer,
        loss=loss.compute
    )

    if args.checkpoint is not None:
        assert os.path.exists(args.checkpoint), "checkpoint does not exist"
        model.load_weights(args.checkpoint, by_name=True)

    model.fit(
        x=training_data_generator,
        validation_data=validation_data_generator if args.validation_split is not None else None,
        batch_size=args.batch_size,
        validation_batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[
            ModelCheckpoint(
                filepath=os.path.join(
                    args.output_dir,
                    "cp_{epoch:02d}_loss-{loss:.2f}.h5" if args.validation_split is None else "cp_{epoch:02d}_loss-{loss:.2f}_valloss-{val_loss:.2f}.h5"
                ),
                save_weights_only=True,
                monitor='loss' if args.validation_split is None else 'val_loss',
                mode='min'
            )
        ]
    )

    model.save_weights(os.path.join(args.output_dir, "model.h5"))
