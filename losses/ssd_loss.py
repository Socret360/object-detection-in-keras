import tensorflow as tf
from .smooth_l1_loss import SMOOTH_L1_LOSS
from .softmax_loss import SOFTMAX_LOSS


class SSD_LOSS:
    """ Loss function as defined in the SSD paper.

    Args:
        - alpha: 

    Returns:
        - 
    """

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.smooth_l1_loss = SMOOTH_L1_LOSS()
        self.softmax_loss = SOFTMAX_LOSS()

    def compute(self, y_true, y_pred):
        smooth_l1 = self.smooth_l1_loss.compute(y_true, y_pred)
        softmax = self.softmax_loss.compute(y_true, y_pred)
        return tf.reduce_mean(y_pred)
