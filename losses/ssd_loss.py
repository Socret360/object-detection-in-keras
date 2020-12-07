import tensorflow as tf


class SSD_LOSS:
    def __init__(self, alpha=1, iou_threshold=0.5):
        self.alpha = alpha
        self.iou_threshold = iou_threshold

    def compute(self, y_true, y_pred):
        return tf.reduce_mean(y_pred)
