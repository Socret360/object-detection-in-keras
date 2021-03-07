from utils import data_utils
from data_generators import TBPP_DATA_GENERATOR, SSD_DATA_GENERATOR
from tensorflow.keras.applications import vgg16, mobilenet_v2, mobilenet


def get_data_generator(config, args):
    training_data_generator, validation_data_generator = None, None
    num_training_samples, num_validation_samples = None, None
    model_config = config["model"]

    training_samples = data_utils.get_samples_from_split(
        split_file=args.training_split,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir
    )
    num_training_samples = len(training_samples)

    if args.validation_split is not None:
        validation_samples = data_utils.get_samples_from_split(
            split_file=args.training_split,
            images_dir=args.images_dir,
            labels_dir=args.labels_dir
        )
        num_validation_samples = len(validation_samples)

    if model_config["name"] == "ssd_vgg16":
        print("creating data generator for tbpp_vgg16")
        training_data_generator = SSD_DATA_GENERATOR(
            samples=training_samples,
            config=config,
            label_maps=args.label_maps,
            shuffle=args.shuffle,
            batch_size=args.batch_size,
            augment=args.augment,
            process_input_fn=vgg16.preprocess_input
        )

        if args.validation_split is not None:
            print("-- validation split specified")
            validation_data_generator = SSD_DATA_GENERATOR(
                samples=validation_samples,
                config=config,
                label_maps=args.label_maps,
                shuffle=args.shuffle,
                batch_size=args.batch_size,
                augment=args.augment,
                process_input_fn=vgg16.preprocess_input
            )
    elif model_config["name"] == "ssd_mobilenetv1":
        print("creating data generator for tbpp_vgg16")
        training_data_generator = SSD_DATA_GENERATOR(
            samples=training_samples,
            config=config,
            label_maps=args.label_maps,
            shuffle=args.shuffle,
            batch_size=args.batch_size,
            augment=args.augment,
            process_input_fn=mobilenet.preprocess_input
        )

        if args.validation_split is not None:
            print("-- validation split specified")
            validation_data_generator = SSD_DATA_GENERATOR(
                samples=validation_samples,
                config=config,
                label_maps=args.label_maps,
                shuffle=args.shuffle,
                batch_size=args.batch_size,
                augment=args.augment,
                process_input_fn=mobilenet.preprocess_input
            )
    elif model_config["name"] == "ssd_mobilenetv2":
        print("creating data generator for tbpp_vgg16")
        training_data_generator = SSD_DATA_GENERATOR(
            samples=training_samples,
            config=config,
            label_maps=args.label_maps,
            shuffle=args.shuffle,
            batch_size=args.batch_size,
            augment=args.augment,
            process_input_fn=mobilenet_v2.preprocess_input
        )

        if args.validation_split is not None:
            print("-- validation split specified")
            validation_data_generator = SSD_DATA_GENERATOR(
                samples=validation_samples,
                config=config,
                label_maps=args.label_maps,
                shuffle=args.shuffle,
                batch_size=args.batch_size,
                augment=args.augment,
                process_input_fn=mobilenet_v2.preprocess_input
            )
    elif model_config["name"] == "tbpp_vgg16":
        print("creating data generator for tbpp_vgg16")
        training_data_generator = TBPP_DATA_GENERATOR(
            samples=training_samples,
            config=config,
            shuffle=args.shuffle,
            batch_size=args.batch_size,
            augment=args.augment,
            process_input_fn=vgg16.preprocess_input
        )

        if args.validation_split is not None:
            print("-- validation split specified")
            validation_data_generator = TBPP_DATA_GENERATOR(
                samples=validation_samples,
                config=config,
                shuffle=args.shuffle,
                batch_size=args.batch_size,
                augment=args.augment,
                process_input_fn=vgg16.preprocess_input
            )
    else:
        print(f"model with name ${model_config['name']} has not been implemented yet")
        exit()

    return training_data_generator, num_training_samples, validation_data_generator, num_validation_samples
