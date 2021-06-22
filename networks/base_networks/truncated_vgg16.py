from custom_layers.l2_normalization import L2Normalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPool2D, Conv2D, Reshape, Concatenate, Activation, Input, ZeroPadding2D, MaxPooling2D
from tensorflow.python.keras.utils import data_utils

WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                       'keras-applications/vgg16/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


def TRUNCATED_VGG16(
    input_shape=None,
    kernel_initializer=None,
    kernel_regularizer=None,
):
    """ A truncated version of VGG16 configuration D
    """
    input_layer = Input(shape=input_shape, name="input")
    # block 1
    conv1_1 = Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1',
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input_layer)
    conv1_2 = Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2',
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', padding="same")(conv1_2)

    # block 2
    conv2_1 = Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1',
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(pool1)
    conv2_2 = Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2',
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', padding="same")(conv2_2)

    # block 3
    conv3_1 = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1',
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(pool2)
    conv3_2 = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2',
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(conv3_1)
    conv3_3 = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3',
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding="same")(conv3_3)

    # block 4
    conv4_1 = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1',
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(pool3)
    conv4_2 = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2',
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(conv4_1)
    conv4_3 = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3',
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', padding="same")(conv4_3)

    # block 5
    conv5_1 = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1',
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(pool4)
    conv5_2 = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2',
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(conv5_1)
    conv5_3 = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3',
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(conv5_2)

    model = Model(inputs=input_layer, outputs=conv5_3)

    weights_path = data_utils.get_file(
        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        file_hash='6d6bbae143d832006294945121d1f1fc')

    model.load_weights(weights_path, by_name=True)

    return model
