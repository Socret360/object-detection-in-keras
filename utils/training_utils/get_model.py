from utils import data_utils
from networks import SSD_MOBILENET, SSD_MOBILENETV2, SSD_VGG16, TBPP_VGG16, QSSD_VGG16, QSSD_MOBILENETV2


def get_model(config, args):
    model_config = config["model"]
    if model_config["name"] == "ssd_vgg16":
        with open(args.label_maps, "r") as label_map_file:
            label_maps = [i.strip("\n") for i in label_map_file.readlines()]

        return SSD_VGG16(
            config=config,
            label_maps=label_maps,
            is_training=True
        )
    elif model_config["name"] == "ssd_mobilenetv1":
        with open(args.label_maps, "r") as label_map_file:
            label_maps = [i.strip("\n") for i in label_map_file.readlines()]

        return SSD_MOBILENET(
            config=config,
            label_maps=label_maps,
            is_training=True
        )
    elif model_config["name"] == "ssd_mobilenetv2":
        with open(args.label_maps, "r") as label_map_file:
            label_maps = [i.strip("\n") for i in label_map_file.readlines()]

        return SSD_MOBILENETV2(
            config=config,
            label_maps=label_maps,
            is_training=True
        )
    elif model_config["name"] == "tbpp_vgg16":
        return TBPP_VGG16(
            config=config,
            is_training=True
        )
    elif model_config["name"] == "qssd_vgg16":
        with open(args.label_maps, "r") as label_map_file:
            label_maps = [i.strip("\n") for i in label_map_file.readlines()]
        model = QSSD_VGG16(
            config=config,
            label_maps=label_maps,
            is_training=True
        )

        return model
    elif model_config["name"] == "qssd_mobilenetv2":
        with open(args.label_maps, "r") as label_map_file:
            label_maps = [i.strip("\n") for i in label_map_file.readlines()]
        model = QSSD_MOBILENETV2(
            config=config,
            label_maps=label_maps,
            is_training=True
        )

        return model
    else:
        print(
            f"model with name ${model_config['name']} has not been implemented yet")
        exit()
