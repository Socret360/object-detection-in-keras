import tensorflow as tf


class SOFTMAX_LOSS:
    """
    Code Reference:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_loss_function/keras_ssd_loss.py
    """

    def compute(self, y_true, y_pred):
        loss = y_true * tf.math.log(y_pred)
        return -1 * tf.reduce_sum(loss, axis=-1)
