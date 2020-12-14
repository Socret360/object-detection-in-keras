import tensorflow as tf


class SMOOTH_L1_LOSS:
    def compute(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * tf.square(y_true - y_pred)
        loss = tf.where(abs_loss < 1, square_loss, abs_loss - 0.5)
        return tf.reduce_sum(loss, axis=-1)
