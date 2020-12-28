import tensorflow as tf


class SMOOTH_L1_LOSS:
    """ Compute smooth l1 loss between the predicted bounding boxes and the ground truth bounding boxes.

    Args:
        - y_true: The ground truth bounding boxes.
        - y_pred: The predicted bounding boxes.

    Code References:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_loss_function/keras_ssd_loss.py

    Paper References:
        - Girshick, R. (2015). Fast-RCNN. https://arxiv.org/pdf/1504.08083.pdf
        - Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y., & Berg, A. C. (2016).
          SSD: Single Shot MultiBox Detector. https://arxiv.org/abs/1512.02325
    """

    def compute(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        res = tf.where(tf.less(abs_loss, 1.0), square_loss, abs_loss - 0.5)
        return tf.reduce_sum(res, axis=-1)
