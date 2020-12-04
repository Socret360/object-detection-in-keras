from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPool2D, Conv2D, Reshape, Concatenate, Activation
from tensorflow.keras.regularizers import l2
from base_networks import VGG16_D
from custom_layers import L2Normalization, DefaultBoxes
from utils import get_number_default_boxes


def SSD300_VGG16_ORIGINAL():
    """ This network follows the official caffe implementation of SSD: https://github.com/chuanqi305/ssd
    1. The authors made a few changes to VGG16 config D layers:
        - fc6 and fc7 is converted into convolutional layers instead of fully connected layers specify in the VGG paper
        - atrous convolution is used to turn fc6 and fc7 into convolutional layers
        - pool5 size is changed from (2, 2) to (3, 3) and its strides is changed from (2, 2) to (1, 1)
        - l2 normalization is used only on the output of conv4_3 because it has different scales compared to other layers. To learn more read SSD paper section 3.1 PASCAL VOC2007
    2. In Keras:
        - padding "same" is equivalent to padding 1 in caffe
        - padding "valid" is equivalent to padding 0 (no padding) in caffe
        - Atrous Convolution is referred to as dilated convolution in Keras and can be used by specifying dilation rate in Conv2D
    """

    # initialize vgg16 config d as base network
    num_classes = 20
    batch_size = None
    l2_reg = 0.0005
    normalize_coords = True
    extra_default_box_for_ar_1 = True
    default_boxes_config = {
        "conv4_3": {
            "aspect_ratios": [1.0, 2.0, 0.5],
            "scale": 0.2,
            "next_scale": 0.31666667,
        },
        "fc7":  {
            "aspect_ratios": [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
            "scale": 0.31666667,
            "next_scale": 0.43333333,
        },
        "conv8_2":  {
            "aspect_ratios": [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
            "scale": 0.43333333,
            "next_scale": 0.55,
        },
        "conv9_2":  {
            "aspect_ratios": [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
            "scale": 0.55,
            "next_scale": 0.66666667,
        },
        "conv10_2": {
            "aspect_ratios": [1.0, 2.0, 0.5],
            "scale": 0.66666667,
            "next_scale": 0.78333333,
        },
        "conv11_2": {
            "aspect_ratios": [1.0, 2.0, 0.5],
            "scale": 0.78333333,
            "next_scale": 0.9,
        },
    }
    input_shape = (300, 300, 3)
    base_network = VGG16_D(num_classes=num_classes, batch_size=batch_size, input_shape=input_shape)
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
    pool5 = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same", name="pool5")(base_network.get_layer('conv5_3').output)
    fc6 = Conv2D(1024, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc6')(pool5)
    fc7 = Conv2D(1024, kernel_size=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7')(fc6)
    conv8_1 = Conv2D(256, kernel_size=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_1')(fc7)
    conv8_2 = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)
    conv9_1 = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)
    conv10_1 = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv10_1')(conv9_2)
    conv10_2 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv10_2')(conv10_1)
    conv11_1 = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv11_1')(conv10_2)
    conv11_2 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv11_2')(conv11_1)
    # normalize conv4_3 as mentioned in the paper
    conv4_3_norm = L2Normalization(gamma_init=20, name="conv4_3_norm")(base_network.get_layer('conv4_3').output)

    # calculate number of default boxes required for each feature map layers
    conv4_3_num_default_boxes = get_number_default_boxes(default_boxes_config['conv4_3']["aspect_ratios"], extra_box_for_ar_1=extra_default_box_for_ar_1)
    fc7_num_default_boxes = get_number_default_boxes(default_boxes_config['fc7']["aspect_ratios"], extra_box_for_ar_1=extra_default_box_for_ar_1)
    conv8_2_num_default_boxes = get_number_default_boxes(default_boxes_config['conv8_2']["aspect_ratios"], extra_box_for_ar_1=extra_default_box_for_ar_1)
    conv9_2_num_default_boxes = get_number_default_boxes(default_boxes_config['conv9_2']["aspect_ratios"], extra_box_for_ar_1=extra_default_box_for_ar_1)
    conv10_2_num_default_boxes = get_number_default_boxes(default_boxes_config['conv10_2']["aspect_ratios"], extra_box_for_ar_1=extra_default_box_for_ar_1)
    conv11_2_num_default_boxes = get_number_default_boxes(default_boxes_config['conv11_2']["aspect_ratios"], extra_box_for_ar_1=extra_default_box_for_ar_1)

    # generate default boxes
    conv4_3_norm_default_boxes = DefaultBoxes(
        image_shape=input_shape,
        scale=default_boxes_config['conv4_3']["scale"],
        next_scale=default_boxes_config['conv4_3']["next_scale"],
        aspect_ratios=default_boxes_config['conv4_3']["aspect_ratios"],
        normalize_coords=normalize_coords,
        name="conv4_3_norm_default_boxes")(base_network.get_layer('conv4_3').output)
    fc7_default_boxes = DefaultBoxes(
        image_shape=input_shape,
        scale=default_boxes_config['fc7']["scale"],
        next_scale=default_boxes_config['fc7']["next_scale"],
        aspect_ratios=default_boxes_config['fc7']["aspect_ratios"],
        normalize_coords=normalize_coords,
        name="fc7_default_boxes")(fc7)
    conv8_2_default_boxes = DefaultBoxes(
        image_shape=input_shape,
        scale=default_boxes_config['conv8_2']["scale"],
        next_scale=default_boxes_config['conv8_2']["next_scale"],
        aspect_ratios=default_boxes_config['conv8_2']["aspect_ratios"],
        normalize_coords=normalize_coords,
        name="conv8_2_default_boxes")(conv8_2)
    conv9_2_default_boxes = DefaultBoxes(
        image_shape=input_shape,
        scale=default_boxes_config['conv9_2']["scale"],
        next_scale=default_boxes_config['conv9_2']["next_scale"],
        aspect_ratios=default_boxes_config['conv9_2']["aspect_ratios"],
        normalize_coords=normalize_coords,
        name="conv9_2_default_boxes")(conv9_2)
    conv10_2_default_boxes = DefaultBoxes(
        image_shape=input_shape,
        scale=default_boxes_config['conv10_2']["scale"],
        next_scale=default_boxes_config['conv10_2']["next_scale"],
        aspect_ratios=default_boxes_config['conv10_2']["aspect_ratios"],
        normalize_coords=normalize_coords,
        name="conv10_2_default_boxes")(conv10_2)
    conv11_2_default_boxes = DefaultBoxes(
        image_shape=input_shape,
        scale=default_boxes_config['conv11_2']["scale"],
        next_scale=default_boxes_config['conv11_2']["next_scale"],
        aspect_ratios=default_boxes_config['conv11_2']["aspect_ratios"],
        normalize_coords=normalize_coords,
        name="conv11_2_default_boxes")(conv11_2)

    # predict class confidence
    conv4_3_norm_mbox_conf = Conv2D(
        filters=conv4_3_num_default_boxes * num_classes,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg),
        name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(
        filters=fc7_num_default_boxes * num_classes,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg),
        name='fc7_mbox_conf')(fc7)
    conv8_2_mbox_conf = Conv2D(
        filters=conv8_2_num_default_boxes * num_classes,
        kernel_size=(3, 3),
        padding='same', kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg),
        name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(
        filters=conv9_2_num_default_boxes * num_classes,
        kernel_size=(3, 3),
        padding='same', kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg),
        name='conv9_2_mbox_conf')(conv9_2)
    conv10_2_mbox_conf = Conv2D(
        filters=conv10_2_num_default_boxes * num_classes,
        kernel_size=(3, 3),
        padding='same', kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg),
        name='conv10_2_mbox_conf')(conv10_2)
    conv11_2_mbox_conf = Conv2D(
        filters=conv11_2_num_default_boxes * num_classes,
        kernel_size=(3, 3),
        padding='same', kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg),
        name='conv11_2_mbox_conf')(conv11_2)

    # predict object location
    conv4_3_norm_mbox_loc = Conv2D(
        filters=conv4_3_num_default_boxes * 4,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg),
        name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(
        filters=fc7_num_default_boxes * 4,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg),
        name='fc7_mbox_loc')(fc7)
    conv8_2_mbox_loc = Conv2D(
        filters=conv8_2_num_default_boxes * 4,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg),
        name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(
        filters=conv9_2_num_default_boxes * 4,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg),
        name='conv9_2_mbox_loc')(conv9_2)
    conv10_2_mbox_loc = Conv2D(
        filters=conv10_2_num_default_boxes * 4,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg),
        name='conv10_2_mbox_loc')(conv10_2)
    conv11_2_mbox_loc = Conv2D(
        filters=conv11_2_num_default_boxes * 4,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(l2_reg),
        name='conv11_2_mbox_loc')(conv11_2)

    # reshape class confidence predictions
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, num_classes), name="conv4_3_norm_mbox_conf_reshape")(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, num_classes), name="fc7_mbox_conf_reshape")(fc7_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, num_classes), name="conv8_2_mbox_conf_reshape")(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, num_classes), name="conv9_2_mbox_conf_reshape")(conv9_2_mbox_conf)
    conv10_2_mbox_conf_reshape = Reshape((-1, num_classes), name="conv10_2_mbox_conf_reshape")(conv10_2_mbox_conf)
    conv11_2_mbox_conf_reshape = Reshape((-1, num_classes), name="conv11_2_mbox_conf_reshape")(conv11_2_mbox_conf)

    # reshape object location predictions
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name="conv4_3_norm_mbox_loc_reshape")(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name="fc7_mbox_loc_reshape")(fc7_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name="conv8_2_mbox_loc_reshape")(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name="conv9_2_mbox_loc_reshape")(conv9_2_mbox_loc)
    conv10_2_mbox_loc_reshape = Reshape((-1, 4), name="conv10_2_mbox_loc_reshape")(conv10_2_mbox_loc)
    conv11_2_mbox_loc_reshape = Reshape((-1, 4), name="conv11_2_mbox_loc_reshape")(conv11_2_mbox_loc)

    # reshape default boxes
    conv4_3_norm_default_boxes_reshape = Reshape((-1, 8), name="conv4_3_norm_default_boxes_reshape")(conv4_3_norm_default_boxes)
    fc7_default_boxes_reshape = Reshape((-1, 8), name="fc7_default_boxes_reshape")(fc7_default_boxes)
    conv8_2_default_boxes_reshape = Reshape((-1, 8), name="conv8_2_default_boxes_reshape")(conv8_2_default_boxes)
    conv9_2_default_boxes_reshape = Reshape((-1, 8), name="conv9_2_default_boxes_reshape")(conv9_2_default_boxes)
    conv10_2_default_boxes_reshape = Reshape((-1, 8), name="conv10_2_default_boxes_reshape")(conv10_2_default_boxes)
    conv11_2_default_boxes_reshape = Reshape((-1, 8), name="conv11_2_default_boxes_reshape")(conv11_2_default_boxes)

    # concentenate class confidence predictions from different feature map layers
    mbox_conf = Concatenate(axis=-2, name="mbox_conf")([
        conv4_3_norm_mbox_conf_reshape,
        fc7_mbox_conf_reshape,
        conv8_2_mbox_conf_reshape,
        conv9_2_mbox_conf_reshape,
        conv10_2_mbox_conf_reshape,
        conv11_2_mbox_conf_reshape
    ])
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # concentenate object location predictions from different feature map layers
    mbox_loc = Concatenate(axis=-2, name="mbox_loc")([
        conv4_3_norm_mbox_loc_reshape,
        fc7_mbox_loc_reshape,
        conv8_2_mbox_loc_reshape,
        conv9_2_mbox_loc_reshape,
        conv10_2_mbox_loc_reshape,
        conv11_2_mbox_loc_reshape
    ])

    # concentenate default boxes from different feature map layers
    mbox_default_boxes = Concatenate(axis=-2, name="mbox_default_boxes")([
        conv4_3_norm_default_boxes_reshape,
        fc7_default_boxes_reshape,
        conv8_2_default_boxes_reshape,
        conv9_2_default_boxes_reshape,
        conv10_2_default_boxes_reshape,
        conv11_2_default_boxes_reshape
    ])

    # concatenate confidence score predictions, bounding box predictions, and default boxes
    predictions = Concatenate(axis=-1, name='predictions')([
        mbox_conf_softmax,
        mbox_loc,
        mbox_default_boxes
    ])

    model = Model(inputs=base_network.input, outputs=predictions)
    model.summary()
