from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten


def VGG16_D(num_classes, batch_size, input_shape):
    input_layer = Input(shape=input_shape, batch_size=batch_size, name="input_1")
    block1_conv1 = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same', name='block1_conv1')(input_layer)
    block1_conv2 = Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same', name='block1_conv2')(block1_conv1)
    block1_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="block1_pool")(block1_conv2)
    block2_conv1 = Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same', name='block2_conv1')(block1_pool)
    block2_conv2 = Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same', name='block2_conv2')(block2_conv1)
    block2_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="block2_pool")(block2_conv2)
    block3_conv1 = Conv2D(256, kernel_size=(3, 3), activation="relu", padding='same', name='block3_conv1')(block2_pool)
    block3_conv2 = Conv2D(256, kernel_size=(3, 3), activation="relu", padding='same', name='block3_conv2')(block3_conv1)
    block3_conv3 = Conv2D(256, kernel_size=(3, 3), activation="relu", padding='same', name='block3_conv3')(block3_conv2)
    block3_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="block3_pool")(block3_conv3)
    block4_conv1 = Conv2D(512, kernel_size=(3, 3), activation="relu", padding='same', name='block4_conv1')(block3_pool)
    block4_conv2 = Conv2D(512, kernel_size=(3, 3), activation="relu", padding='same', name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(512, kernel_size=(3, 3), activation="relu", padding='same', name='block4_conv3')(block4_conv2)
    block4_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="block4_pool")(block4_conv3)
    block5_conv1 = Conv2D(512, kernel_size=(3, 3), activation="relu", padding='same', name='block5_conv1')(block4_pool)
    block5_conv2 = Conv2D(512, kernel_size=(3, 3), activation="relu", padding='same', name='block5_conv2')(block5_conv1)
    block5_conv3 = Conv2D(512, kernel_size=(3, 3), activation="relu", padding='same', name='block5_conv3')(block5_conv2)
    block5_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="block5_pool")(block5_conv3)
    flatten = Flatten(name="flatten")(block5_pool)
    fc1 = Dense(4096, activation="relu", name="fc1")(flatten)
    fc2 = Dense(4096, activation="relu", name="fc2")(fc1)
    predictions = Dense(num_classes, activation="softmax", name="predictions")(fc2)
    return Model(inputs=input_layer, outputs=predictions)
