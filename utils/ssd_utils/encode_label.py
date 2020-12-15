import numpy as np


def encode_label(y, epsilon=10e-5):
    """ Encode the label to a proper format suitable for training SSD network.

    Args:
        - y: A numpy of shape (num_default_boxes, num_classes + 12) representing a label sample.

    Returns:
        - A numpy array with the same shape as y but its gt boxes values has been encoded to the proper SSD format.

    Paper References:
        - Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y., & Berg, A. C. (2016).
          SSD: Single Shot MultiBox Detector. https://arxiv.org/abs/1512.02325

    Webpage References:
        - https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/

    Code References:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd_encoder_decoder/ssd_input_encoder.py
    """
    gt_boxes = y[:, -12:-8]
    df_boxes = y[:, -8:-4]
    variances = y[:, -4:]
    encoded_gt_boxes_cx = ((gt_boxes[:, 0] - df_boxes[:, 0]) / (df_boxes[:, 2])) / np.sqrt(variances[:, 0])
    encoded_gt_boxes_cy = ((gt_boxes[:, 1] - df_boxes[:, 1]) / (df_boxes[:, 3])) / np.sqrt(variances[:, 1])
    encoded_gt_boxes_w = np.log(epsilon + gt_boxes[:, 2] / df_boxes[:, 2]) / np.sqrt(variances[:, 2])
    encoded_gt_boxes_h = np.log(epsilon + gt_boxes[:, 3] / df_boxes[:, 3]) / np.sqrt(variances[:, 3])
    y[:, -12] = encoded_gt_boxes_cx
    y[:, -11] = encoded_gt_boxes_cy
    y[:, -10] = encoded_gt_boxes_w
    y[:, -9] = encoded_gt_boxes_h
    return y
