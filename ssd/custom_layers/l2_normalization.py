import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class L2Normalization(Layer):
    """ A custom layer that performs l2 normalization on its inputs with learnable parameter gamma.
    Note:
    1. This is implementation is taken from https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_layers/keras_layer_L2Normalization.py with slight modifications:
        - axis variable is passed as parameter instead of fixed value
        - K.variable is replaced with tf.Variable
        - fixed dtype mismatched by specifying dtype=np.float32
    2. get_config & from_config is necessary to make the layer serializable
    3. we need to multiply self.gamma_init with np.ones((input_shape[self.axis],), dtype=np.float32)
       to turn gamma into the shape of (input_shape[axis],) which will allow us to broadcast those values
       when multiplying with the output in the call function.
    """

    def __init__(self, gamma_init=20, axis=-1, **kwargs):
        self.axis = axis
        self.gamma_init = gamma_init
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        gamma = self.gamma_init * np.ones((input_shape[self.axis],), dtype=np.float32)
        self.gamma = tf.Variable(gamma, name=f"{self.name}_gamma", trainable=True)
        super(L2Normalization, self).build(input_shape)

    def call(self, inputs):
        output = K.l2_normalize(inputs, self.axis)
        return output * self.gamma

    def get_config(self):
        config = {'gamma_init': self.gamma_init, 'axis': self.axis}
        base_config = super(L2Normalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
