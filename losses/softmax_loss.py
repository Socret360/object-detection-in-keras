import tensorflow as tf


class SOFTMAX_LOSS:
    """ Calculates the softmax loss between the predicted classes and ground truth classes.

    Args:
        - y_true: The ground truth classes.
        - y_pred: The predicted classes.

    Code Reference:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_loss_function/keras_ssd_loss.py
    """

    def compute(self, y_true, y_pred):
        y_pred = tf.maximum(y_pred, 1e-15)
        return -1 * tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
