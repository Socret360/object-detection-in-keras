import numpy as np
import tensorflow as tf
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
       to turn gamma into the shape of (input_shape[self.axis],) which will allow us to broadcast those values
       when multiplying with the output in the call function.

    Args:
        - gamma_init: The initial scaling parameter. Defaults to 20 following the SSD paper.
        - axis: the axis to apply the scaling to

    Returns:
        - A scaled tensor with the same shape as input_shape

    Code References:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_layers/keras_layer_L2Normalization.py

    Paper References:
        - Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C. Y., & Berg, A. C. (2016).
          SSD: Single Shot MultiBox Detector. https://arxiv.org/abs/1512.02325
        - Liu, W., Rabinovich, A., & Berg, A. C. (2016).
          ParseNet: Looking Wider to See Better. International Conference on Learning Representation (ICLR) 2016.
          https://arxiv.org/abs/1506.04579
    """

    def __init__(self, gamma_init=20, axis=-1, **kwargs):
        self.axis = axis
        self.gamma_init = gamma_init
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        gamma = self.gamma_init * np.ones((input_shape[self.axis],), dtype=np.float32)
        self.gamma = tf.Variable(gamma, trainable=True)
        super(L2Normalization, self).build(input_shape)

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, self.axis) * self.gamma

    def get_config(self):
        config = {'gamma_init': self.gamma_init, 'axis': self.axis}
        base_config = super(L2Normalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
