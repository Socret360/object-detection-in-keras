from utils import data_utils
from losses import SSD_LOSS, TBPP_LOSS


def get_loss(config, args):
    model_config = config["model"]
    training_config = config["training"]
    if model_config["name"] == "ssd_vgg16":
        return SSD_LOSS(
            alpha=training_config["alpha"],
            min_negative_boxes=training_config["min_negative_boxes"],
            negative_boxes_ratio=training_config["negative_boxes_ratio"]
        )
    elif model_config["name"] == "ssd_mobilenetv1":
        return SSD_LOSS(
            alpha=training_config["alpha"],
            min_negative_boxes=training_config["min_negative_boxes"],
            negative_boxes_ratio=training_config["negative_boxes_ratio"]
        )
    elif model_config["name"] == "ssd_mobilenetv2":
        return SSD_LOSS(
            alpha=training_config["alpha"],
            min_negative_boxes=training_config["min_negative_boxes"],
            negative_boxes_ratio=training_config["negative_boxes_ratio"]
        )
    elif model_config["name"] == "tbpp_vgg16":
        return TBPP_LOSS(
            alpha=training_config["alpha"],
            min_negative_boxes=training_config["min_negative_boxes"],
            negative_boxes_ratio=training_config["negative_boxes_ratio"]
        )
    else:
        print(f"model with name ${model_config['name']} has not been implemented yet")
        exit()
