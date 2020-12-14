import tensorflow as tf


class SOFTMAX_LOSS:
    def compute(self, y_true, y_pred):
        loss = y_true * tf.math.log(y_pred)
        return -1 * tf.reduce_sum(loss, axis=-1)
