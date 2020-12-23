import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPool2D, Conv2D, Reshape, Concatenate, Activation
from tensorflow.keras.regularizers import l2
from base_networks import VGG16_D
from custom_layers import L2Normalization, DefaultBoxes
from utils.ssd_utils import get_number_default_boxes


def SSD300_VGG16(config):
    """ This network follows the official caffe implementation of SSD: https://github.com/chuanqi305/ssd
    1. Changes made to VGG16 config D layers:
        - fc6 and fc7 is converted into convolutional layers instead of fully connected layers specify in the VGG paper
        - atrous convolution is used to turn fc6 and fc7 into convolutional layers
        - pool5 size is changed from (2, 2) to (3, 3) and its strides is changed from (2, 2) to (1, 1)
        - l2 normalization is used only on the output of conv4_3 because it has different scales compared to other layers. To learn more read SSD paper section 3.1 PASCAL VOC2007
    2. In Keras:
        - padding "same" is equivalent to padding 1 in caffe
        - padding "valid" is eque authors made a few ch ivalent to padding 0 (no padding) in caffe
        - Atrous Convolution is referred to as dilated convolution in Keras and can be used by specifying dilation rate in Conv2D
    3. The name of each layer in the network is renamed to match the official caffe implementation

    Args:
        - config: python dict as read from the config file

    Returns:
        - A keras version of SSD300 with VGG16 as backbone network.

    Code References:
        - https://github.com/chuanqi305/ssd
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/models/keras_ssd300.py

    Paper References:
        - Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C. Y., & Berg, A. C. (2016).
          SSD: Single Shot MultiBox Detector. https://arxiv.org/abs/1512.02325
    """
    input_shape = config["model"]["input_shape"]
    num_classes = config["model"]["num_classes"] + 1  # for background class
    l2_reg = config["model"]["l2_regularization"]
    kernel_initializer = config["model"]["kernel_initializer"]
    default_boxes_config = config["model"]["default_boxes"]
    extra_box_for_ar_1 = config["model"]["extra_box_for_ar_1"]

    # construct the base network and extra feature layers
    base_network = VGG16_D(num_classes=num_classes, input_shape=input_shape)
    base_network = Model(inputs=base_network.input, outputs=base_network.get_layer('block5_conv3').output)
    base_network.get_layer("input_1")._name = "input"
    for layer in base_network.layers:
        if "pool" in layer.name:
            new_name = layer.name.replace("block", "")
            new_name = new_name.split("_")
            new_name = f"{new_name[1]}{new_name[0]}"
        else:
            new_name = layer.name.replace("conv", "")
            new_name = new_name.replace("block", "conv")
        base_network.get_layer(layer.name)._name = new_name
        base_network.get_layer(layer.name)._kernel_initializer = "he_normal"
        base_network.get_layer(layer.name)._kernel_regularizer = l2(l2_reg)
        layer.trainable = False  # each layer of the base network should not be trainable

    pool5 = MaxPool2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="pool5")(base_network.get_layer('conv5_3').output)
    fc6 = Conv2D(
        1024,
        kernel_size=(3, 3),
        dilation_rate=(6, 6),
        activation='relu',
        padding='same',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2(l2_reg),
        name='fc6')(pool5)
    fc7 = Conv2D(
        1024,
        kernel_size=(1, 1),
        activation='relu',
        padding='same',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2(l2_reg),
        name='fc7')(fc6)
    conv8_1 = Conv2D(
        256,
        kernel_size=(1, 1),
        activation='relu',
        padding='valid',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2(l2_reg),
        name='conv8_1')(fc7)
    conv8_2 = Conv2D(
        512,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation='relu',
        padding='same',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2(l2_reg),
        name='conv8_2')(conv8_1)
    conv9_1 = Conv2D(
        128,
        kernel_size=(1, 1),
        activation='relu',
        padding='valid',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2(l2_reg),
        name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(
        256,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation='relu',
        padding='same',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2(l2_reg),
        name='conv9_2')(conv9_1)
    conv10_1 = Conv2D(
        128,
        kernel_size=(1, 1),
        activation='relu',
        padding='valid',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2(l2_reg),
        name='conv10_1')(conv9_2)
    conv10_2 = Conv2D(
        256,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation='relu',
        padding='valid',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2(l2_reg),
        name='conv10_2')(conv10_1)
    conv11_1 = Conv2D(
        128,
        kernel_size=(1, 1),
        activation='relu',
        padding='valid',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2(l2_reg),
        name='conv11_1')(conv10_2)
    conv11_2 = Conv2D(
        256,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation='relu',
        padding='valid',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2(l2_reg),
        name='conv11_2')(conv11_1)
    model = Model(inputs=base_network.input, outputs=conv11_2)

    # construct the prediction layers (conf, loc, & default_boxes)
    scales = np.linspace(
        default_boxes_config["min_scale"],
        default_boxes_config["max_scale"],
        len(default_boxes_config["layers"])
    )
    mbox_conf_layers = []
    mbox_loc_layers = []
    mbox_default_boxes_layers = []
    for i, layer in enumerate(default_boxes_config["layers"]):
        num_default_boxes = get_number_default_boxes(
            layer["aspect_ratios"],
            extra_box_for_ar_1=extra_box_for_ar_1
        )
        x = model.get_layer(layer["name"]).output
        layer_name = layer["name"]

        # conv4_3 has different scales compared to other feature map layers
        if layer_name == "conv4_3":
            layer_name = f"{layer_name}_norm"
            x = L2Normalization(gamma_init=20, name=layer_name)(x)

        layer_mbox_conf = Conv2D(
            filters=num_default_boxes * num_classes,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=f"{layer_name}_mbox_conf")(x)
        layer_mbox_conf_reshape = Reshape((-1, num_classes), name=f"{layer_name}_mbox_conf_reshape")(layer_mbox_conf)
        layer_mbox_loc = Conv2D(
            filters=num_default_boxes * 4,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=f"{layer_name}_mbox_loc")(x)
        layer_mbox_loc_reshape = Reshape((-1, 4), name=f"{layer_name}_mbox_loc_reshape")(layer_mbox_loc)
        layer_default_boxes = DefaultBoxes(
            image_shape=input_shape,
            scale=scales[i],
            next_scale=scales[i+1] if i+1 <= len(default_boxes_config["layers"]) - 1 else 1,
            aspect_ratios=layer["aspect_ratios"],
            variances=default_boxes_config["variances"],
            extra_box_for_ar_1=extra_box_for_ar_1,
            name=f"{layer_name}_default_boxes")(x)
        layer_default_boxes_reshape = Reshape((-1, 8), name=f"{layer_name}_default_boxes_reshape")(layer_default_boxes)
        mbox_conf_layers.append(layer_mbox_conf_reshape)
        mbox_loc_layers.append(layer_mbox_loc_reshape)
        mbox_default_boxes_layers.append(layer_default_boxes_reshape)

    # concentenate class confidence predictions from different feature map layers
    mbox_conf = Concatenate(axis=-2, name="mbox_conf")(mbox_conf_layers)
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)
    # concentenate object location predictions from different feature map layers
    mbox_loc = Concatenate(axis=-2, name="mbox_loc")(mbox_loc_layers)
    # concentenate default boxes from different feature map layers
    mbox_default_boxes = Concatenate(axis=-2, name="mbox_default_boxes")(mbox_default_boxes_layers)
    # concatenate confidence score predictions, bounding box predictions, and default boxes
    predictions = Concatenate(axis=-1, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_default_boxes])
    model = Model(inputs=base_network.input, outputs=predictions)
    return model
