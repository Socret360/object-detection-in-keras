import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPool2D, Conv2D, Reshape, Concatenate, Activation, Input, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from custom_layers import L2Normalization, DefaultBoxes, DecodeSSDPredictions, DecodeTBPPPredictions
from utils.ssd_utils import get_number_default_boxes


def TBPP_VGG16(
    config,
    num_predictions=10,
    is_training=True,
):
    model_config = config["model"]

    if is_training:
        input_shape = (None, None, 3)
    else:
        input_shape = (model_config["input_size"],
                       model_config["input_size"], 3)

    num_classes = 2  # 1 for text and 1 for background
    l2_reg = model_config["l2_regularization"]
    kernel_initializer = model_config["kernel_initializer"]
    default_boxes_config = model_config["default_boxes"]
    extra_box_for_ar_1 = default_boxes_config["extra_box_for_ar_1"]

    input_tensor = Input(shape=input_shape)
    input_tensor = ZeroPadding2D(padding=(2, 2))(input_tensor)

    # construct the base network and extra feature layers
    base_network = VGG16(
        input_tensor=input_tensor,
        classes=num_classes,
        weights='imagenet',
        include_top=False
    )

    base_network = Model(inputs=base_network.input,
                         outputs=base_network.get_layer('block5_conv3').output)
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

    def conv_block_1(x, filters, name, padding='valid', dilation_rate=(1, 1), strides=(1, 1)):
        return Conv2D(
            filters,
            kernel_size=(3, 3),
            strides=strides,
            activation='relu',
            padding=padding,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=name)(x)

    def conv_block_2(x, filters, name, padding='valid', dilation_rate=(1, 1), strides=(1, 1)):
        return Conv2D(
            filters,
            kernel_size=(1, 1),
            strides=strides,
            activation='relu',
            padding=padding,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=name)(x)

    pool5 = MaxPool2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="pool5")(base_network.get_layer('conv5_3').output)

    fc6 = conv_block_1(x=pool5, filters=1024, padding="same",
                       dilation_rate=(6, 6), name="fc6")
    fc7 = conv_block_2(x=fc6, filters=1024, padding="same", name="fc7")
    conv8_1 = conv_block_2(x=fc7, filters=256, padding="valid", name="conv8_1")
    conv8_2 = conv_block_1(x=conv8_1, filters=512,
                           padding="same", strides=(2, 2), name="conv8_2")
    conv9_1 = conv_block_2(x=conv8_2, filters=128,
                           padding="valid", name="conv9_1")
    conv9_2 = conv_block_1(x=conv9_1, filters=256,
                           padding="same", strides=(2, 2), name="conv9_2")
    conv10_1 = conv_block_2(x=conv9_2, filters=128,
                            padding="valid", name="conv10_1")
    conv10_2 = conv_block_1(x=conv10_1, filters=256,
                            padding="valid", name="conv10_2")
    conv11_1 = conv_block_2(x=conv10_2, filters=128,
                            padding="valid", name="conv11_1")
    conv11_2 = conv_block_1(x=conv11_1, filters=256,
                            padding="valid", name="conv11_2")

    model = Model(inputs=base_network.input, outputs=conv11_2)

    # construct the prediction layers (conf, loc, & default_boxes)
    scales = np.linspace(
        default_boxes_config["min_scale"],
        default_boxes_config["max_scale"],
        len(default_boxes_config["layers"])
    )
    mbox_conf_layers = []
    mbox_loc_layers = []
    mbox_quad_layers = []
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
            kernel_size=(3, 5),
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=f"{layer_name}_mbox_conf")(x)
        layer_mbox_conf_reshape = Reshape(
            (-1, num_classes), name=f"{layer_name}_mbox_conf_reshape")(layer_mbox_conf)
        layer_mbox_loc = Conv2D(
            filters=num_default_boxes * 4,
            kernel_size=(3, 5),
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=f"{layer_name}_mbox_loc")(x)
        layer_mbox_loc_reshape = Reshape(
            (-1, 4), name=f"{layer_name}_mbox_loc_reshape")(layer_mbox_loc)
        layer_mbox_quad = Conv2D(
            filters=num_default_boxes * 8,
            kernel_size=(3, 5),
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=f"{layer_name}_mbox_quad")(x)
        layer_mbox_quad_reshape = Reshape(
            (-1, 8), name=f"{layer_name}_mbox_quad_reshape")(layer_mbox_quad)
        mbox_conf_layers.append(layer_mbox_conf_reshape)
        mbox_loc_layers.append(layer_mbox_loc_reshape)
        mbox_quad_layers.append(layer_mbox_quad_reshape)

    # concentenate class confidence predictions from different feature map layers
    mbox_conf = Concatenate(axis=-2, name="mbox_conf")(mbox_conf_layers)
    mbox_conf_softmax = Activation(
        'softmax', name='mbox_conf_softmax')(mbox_conf)
    # concentenate object location predictions from different feature map layers
    mbox_loc = Concatenate(axis=-2, name="mbox_loc")(mbox_loc_layers)
    # concentenate object quad predictions from different feature map layers
    mbox_quad = Concatenate(axis=-2, name="mbox_quad")(mbox_quad_layers)

    if is_training:
        # concatenate confidence score predictions, bounding box predictions, and default boxes
        predictions = Concatenate(
            axis=-1, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_quad])
        return Model(inputs=base_network.input, outputs=predictions)

    mbox_default_boxes_layers = []
    for i, layer in enumerate(default_boxes_config["layers"]):
        num_default_boxes = get_number_default_boxes(
            layer["aspect_ratios"],
            extra_box_for_ar_1=extra_box_for_ar_1
        )
        x = model.get_layer(layer["name"]).output
        layer_name = layer["name"]
        layer_default_boxes = DefaultBoxes(
            image_shape=input_shape,
            scale=scales[i],
            next_scale=scales[i+1] if i +
            1 <= len(default_boxes_config["layers"]) - 1 else 1,
            aspect_ratios=layer["aspect_ratios"],
            variances=default_boxes_config["variances"],
            extra_box_for_ar_1=extra_box_for_ar_1,
            name=f"{layer_name}_default_boxes")(x)
        layer_default_boxes_reshape = Reshape(
            (-1, 8), name=f"{layer_name}_default_boxes_reshape")(layer_default_boxes)
        mbox_default_boxes_layers.append(layer_default_boxes_reshape)

    # concentenate default boxes from different feature map layers
    mbox_default_boxes = Concatenate(
        axis=-2, name="mbox_default_boxes")(mbox_default_boxes_layers)
    predictions = Concatenate(axis=-1, name='predictions')(
        [mbox_conf_softmax, mbox_loc, mbox_quad, mbox_default_boxes])
    decoded_predictions = DecodeTBPPPredictions(
        input_size=model_config["input_size"],
        num_predictions=num_predictions,
        name="decoded_predictions"
    )(predictions)
    return Model(inputs=base_network.input, outputs=decoded_predictions)
