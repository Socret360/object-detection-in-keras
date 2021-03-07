from utils import data_utils
from networks import SSD_MOBILENET, SSD_MOBILENETV2, SSD_VGG16, TBPP_VGG16


def get_model(config, args):
    model_config = config["model"]
    if model_config["name"] == "ssd_vgg16":
        return SSD_VGG16(
            config=config,
            label_maps=args.label_maps,
            is_training=True
        )
    elif model_config["name"] == "ssd_mobilenetv1":
        return SSD_MOBILENET(
            config=config,
            label_maps=args.label_maps,
            is_training=True
        )
    elif model_config["name"] == "ssd_mobilenetv2":
        return SSD_MOBILENETV2(
            config=config,
            label_maps=args.label_maps,
            is_training=True
        )
    elif model_config["name"] == "tbpp_vgg16":
        return TBPP_VGG16(
            config=config,
            is_training=True
        )
    else:
        print(f"model with name ${model_config['name']} has not been implemented yet")
        exit()
