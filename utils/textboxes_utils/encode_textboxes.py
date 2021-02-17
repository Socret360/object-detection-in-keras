import numpy as np
from utils import bbox_utils


def encode_textboxes(y, epsilon=10e-5):
    """ Encode the label to a proper format suitable for training TextBoxes PlusPlus network.

    Args:
        - y: A numpy of shape (num_default_boxes, 2 + 12 + 8) representing a label sample.

    Returns:
        - A numpy array with the same shape as y but its gt boxes values has been encoded to the proper TextBoxes PlusPlus format.

    Paper References:
        - Liao, M., Shi, B., & Bai, X. (2018). TextBoxes++: A Single-Shot Oriented Scene Text Detector. https://arxiv.org/abs/1512.02325
    """
    gt_textboxes = y[:, -20:-8]
    df_boxes = y[:, -8:-4]
    df_boxes_vertices = bbox_utils.center_to_vertices(df_boxes)
    variances = y[:, -4:]
    encoded_gt_textboxes_cx = ((gt_textboxes[:, 0] - df_boxes[:, 0]) / (df_boxes[:, 2])) / np.sqrt(variances[:, 0])
    encoded_gt_textboxes_cy = ((gt_textboxes[:, 1] - df_boxes[:, 1]) / (df_boxes[:, 3])) / np.sqrt(variances[:, 1])
    encoded_gt_textboxes_w = np.log(epsilon + gt_textboxes[:, 2] / df_boxes[:, 2]) / np.sqrt(variances[:, 2])
    encoded_gt_textboxes_h = np.log(epsilon + gt_textboxes[:, 3] / df_boxes[:, 3]) / np.sqrt(variances[:, 3])
    encoded_gt_textboxes_x1 = ((gt_textboxes[:, 4] - df_boxes_vertices[:, 0, 0]) / df_boxes[:, 2]) / np.sqrt(variances[:, 0])
    encoded_gt_textboxes_y1 = ((gt_textboxes[:, 5] - df_boxes_vertices[:, 0, 1]) / df_boxes[:, 3]) / np.sqrt(variances[:, 1])
    encoded_gt_textboxes_x2 = ((gt_textboxes[:, 6] - df_boxes_vertices[:, 1, 0]) / df_boxes[:, 2]) / np.sqrt(variances[:, 0])
    encoded_gt_textboxes_y2 = ((gt_textboxes[:, 7] - df_boxes_vertices[:, 1, 1]) / df_boxes[:, 3]) / np.sqrt(variances[:, 1])
    encoded_gt_textboxes_x3 = ((gt_textboxes[:, 8] - df_boxes_vertices[:, 2, 0]) / df_boxes[:, 2]) / np.sqrt(variances[:, 0])
    encoded_gt_textboxes_y3 = ((gt_textboxes[:, 9] - df_boxes_vertices[:, 2, 1]) / df_boxes[:, 3]) / np.sqrt(variances[:, 1])
    encoded_gt_textboxes_x4 = ((gt_textboxes[:, 10] - df_boxes_vertices[:, 3, 0]) / df_boxes[:, 2]) / np.sqrt(variances[:, 0])
    encoded_gt_textboxes_y4 = ((gt_textboxes[:, 11] - df_boxes_vertices[:, 3, 1]) / df_boxes[:, 3]) / np.sqrt(variances[:, 1])
    y[:, -20] = encoded_gt_textboxes_cx
    y[:, -19] = encoded_gt_textboxes_cy
    y[:, -18] = encoded_gt_textboxes_w
    y[:, -17] = encoded_gt_textboxes_h
    y[:, -16] = encoded_gt_textboxes_x1
    y[:, -15] = encoded_gt_textboxes_y1
    y[:, -14] = encoded_gt_textboxes_x2
    y[:, -13] = encoded_gt_textboxes_y2
    y[:, -12] = encoded_gt_textboxes_x3
    y[:, -11] = encoded_gt_textboxes_y3
    y[:, -10] = encoded_gt_textboxes_x4
    y[:, -9] = encoded_gt_textboxes_y4
    return y
