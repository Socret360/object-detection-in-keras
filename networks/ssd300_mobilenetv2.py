import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Reshape, Concatenate, Activation
from tensorflow.keras.applications import MobileNetV2
from custom_layers import DefaultBoxes, DecodeSSDPredictions
from utils.ssd_utils import get_number_default_boxes


def SSD300_MOBILENET_V2(
    config,
    label_maps,
    num_predictions=10,
    is_training=True,
):
    """ Construct an SSD network that uses MobileNetV1 backbone.

    Args:
        - config: python dict as read from the config file
        - label_maps: A python list containing the classes
        - num_predictions: The number of predictions to produce as final output
        - is_training: whether the model is constructed for training purpose or inference purpose

    Returns:
        - A keras version of SSD300 with MobileNetV2 as backbone network.

    Code References:
        - https://github.com/chuanqi305/MobileNet-SSD
    """
    model_config = config["model"]
    input_shape = (model_config["input_size"], model_config["input_size"], 3)
    num_classes = len(label_maps) + 1  # for background class
    l2_reg = model_config["l2_regularization"]
    kernel_initializer = model_config["kernel_initializer"]
    default_boxes_config = model_config["default_boxes"]
    extra_box_for_ar_1 = model_config["extra_box_for_ar_1"]
    #
    base_network = MobileNetV2(
        input_shape=input_shape,
        alpha=config["model"]["width_multiplier"],
        classes=num_classes,
        weights='imagenet',
        include_top=False
    )
    base_network = Model(inputs=base_network.input, outputs=base_network.get_layer('block_16_project_BN').output)
    base_network.get_layer("input_1")._name = "input"
    # [
    #     {
    #         "name": "block_13_expand_relu",
    #         "size": 19
    #     },
    #     {
    #         "name": "block_16_project_BN",
    #         "size": 10
    #     },
    # ]
    return base_network
