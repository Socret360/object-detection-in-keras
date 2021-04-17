import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPool2D, Conv2D, Reshape, Concatenate, Activation, Input, ZeroPadding2D, BatchNormalization, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2
from custom_layers import DefaultQuads
from utils.qssd_utils import get_number_default_quads


def QSSD_MOBILENETV2(
    config,
    label_maps,
    num_predictions=10,
    is_training=True,
):
    model_config = config["model"]
    input_shape = (model_config["input_size"], model_config["input_size"], 3)
    num_classes = len(label_maps) + 1  # for background class
    l2_reg = model_config["l2_regularization"]
    kernel_initializer = model_config["kernel_initializer"]
    default_quads_config = model_config["default_quads"]
    extra_box_for_ar_1 = model_config["extra_box_for_ar_1"]
    #

    input_tensor = Input(shape=input_shape)
    input_tensor = ZeroPadding2D(padding=(2, 2))(input_tensor)

    base_network = MobileNetV2(
        input_tensor=input_tensor,
        alpha=config["model"]["width_multiplier"],
        classes=num_classes,
        weights='imagenet',
        include_top=False
    )
    base_network = Model(inputs=base_network.input, outputs=base_network.get_layer('block_16_project_BN').output)
    base_network.get_layer("input_1")._name = "input"
    for layer in base_network.layers:
        base_network.get_layer(layer.name)._kernel_initializer = "he_normal"
        base_network.get_layer(layer.name)._kernel_regularizer = l2(l2_reg)
        layer.trainable = False  # each layer of the base network should not be trainable

    conv_13 = base_network.get_layer("block_13_expand_relu").output
    conv_16 = base_network.get_layer('block_16_project_BN').output

    def conv_block_1(x, filters, name):
        x = Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            padding="valid",
            kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg),
            name=name,
            use_bias=False)(x)
        x = BatchNormalization(name=f"{name}/bn")(x)
        x = ReLU(name=f"{name}/relu")(x)
        return x

    def conv_block_2(x, filters, name):
        x = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg),
            name=name,
            use_bias=False,
            strides=(2, 2))(x)
        x = BatchNormalization(name=f"{name}/bn")(x)
        x = ReLU(name=f"{name}/relu")(x)
        return x
    conv17_1 = conv_block_1(x=conv_16, filters=256, name="conv17_1")
    conv17_2 = conv_block_2(x=conv17_1, filters=512, name="conv17_2")
    conv18_1 = conv_block_1(x=conv17_2, filters=128, name="conv18_1")
    conv18_2 = conv_block_2(x=conv18_1, filters=256, name="conv18_2")
    conv19_1 = conv_block_1(x=conv18_2, filters=128, name="conv19_1")
    conv19_2 = conv_block_2(x=conv19_1, filters=256, name="conv19_2")
    conv20_1 = conv_block_1(x=conv19_2, filters=128, name="conv20_1")
    conv20_2 = conv_block_2(x=conv20_1, filters=256, name="conv20_2")
    model = Model(inputs=base_network.input, outputs=conv20_2)

    # construct the prediction layers (conf, loc, & default_boxes)
    scales = np.linspace(
        default_quads_config["min_scale"],
        default_quads_config["max_scale"],
        len(default_quads_config["layers"])
    )
    mbox_conf_layers = []
    mbox_quad_layers = []
    for i, layer in enumerate(default_quads_config["layers"]):
        num_default_quads = get_number_default_quads(
            aspect_ratios=layer["aspect_ratios"],
            angles=layer["angles"],
            extra_box_for_ar_1=extra_box_for_ar_1
        )
        x = model.get_layer(layer["name"]).output
        layer_name = layer["name"]

        layer_mbox_conf = Conv2D(
            filters=num_default_quads * num_classes,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=f"{layer_name}_mbox_conf")(x)
        layer_mbox_conf_reshape = Reshape(
            (-1, num_classes), name=f"{layer_name}_mbox_conf_reshape")(layer_mbox_conf)
        layer_mbox_quad = Conv2D(
            filters=num_default_quads * 8,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(l2_reg),
            name=f"{layer_name}_mbox_quad")(x)
        layer_mbox_quad_reshape = Reshape(
            (-1, 8), name=f"{layer_name}_mbox_quad_reshape")(layer_mbox_quad)
        mbox_conf_layers.append(layer_mbox_conf_reshape)
        mbox_quad_layers.append(layer_mbox_quad_reshape)

    # concentenate class confidence predictions from different feature map layers
    mbox_conf = Concatenate(axis=-2, name="mbox_conf")(mbox_conf_layers)
    mbox_conf_softmax = Activation(
        'softmax', name='mbox_conf_softmax')(mbox_conf)
    # concentenate object quad predictions from different feature map layers
    mbox_quad = Concatenate(axis=-2, name="mbox_quad")(mbox_quad_layers)

    if is_training:
        # concatenate confidence score predictions, bounding box predictions, and default boxes
        predictions = Concatenate(
            axis=-1, name='predictions')([mbox_conf_softmax, mbox_quad])
        return Model(inputs=base_network.input, outputs=predictions)

    mbox_default_quads_layers = []
    for i, layer in enumerate(default_quads_config["layers"]):
        num_default_quads = get_number_default_quads(
            aspect_ratios=layer["aspect_ratios"],
            angles=layer["angles"],
            extra_box_for_ar_1=extra_box_for_ar_1
        )
        x = model.get_layer(layer["name"]).output
        layer_name = layer["name"]
        layer_default_quads = DefaultQuads(
            image_shape=input_shape,
            scale=scales[i],
            next_scale=scales[i+1] if i +
            1 <= len(default_quads_config["layers"]) - 1 else 1,
            aspect_ratios=layer["aspect_ratios"],
            angles=layer["angles"],
            variances=default_quads_config["variances"],
            extra_box_for_ar_1=extra_box_for_ar_1,
            name=f"{layer_name}_default_quads")(x)
        layer_default_quads_reshape = Reshape(
            (-1, 16), name=f"{layer_name}_default_quads_reshape")(layer_default_quads)
        mbox_default_quads_layers.append(layer_default_quads_reshape)

    # concentenate default boxes from different feature map layers
    mbox_default_quads = Concatenate(
        axis=-2, name="mbox_default_quads")(mbox_default_quads_layers)
    predictions = Concatenate(axis=-1, name='predictions')(
        [mbox_conf_softmax, mbox_quad, mbox_default_quads])
    # decoded_predictions = DecodeQSSDPredictions(
    #     input_size=model_config["input_size"],
    #     num_predictions=num_predictions,
    #     name="decoded_predictions"
    # )(predictions)

    return Model(inputs=base_network.input, outputs=predictions)
