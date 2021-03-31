import numpy as np
from utils import bbox_utils


def encode_qboxes(y, epsilon=10e-5):
    """ Encode the label to a proper format suitable for training QSSD network.

    Args:
        - y: A numpy of shape (num_default_boxes, 2 + 12 + 8) representing a label sample.

    Returns:
        - A numpy array with the same shape as y but its gt boxes values has been encoded to the proper QSSD format.

    """
    gt_qboxes = y[:, -20:-8]
    gt_qboxes = bbox_utils.center_to_corner(gt_qboxes)
    df_boxes = y[:, -16:-12]
    df_boxes_cor = bbox_utils.center_to_corner(df_boxes)
    df_boxes_vertices = bbox_utils.center_to_vertices(df_boxes)
    variances = y[:, -12:]
    #
    gt_qboxes_xmin = (gt_qboxes[:, 0] - df_boxes_cor[:, 0]) / df_boxes[:, 2]
    gt_qboxes_xmin /= np.sqrt(variances[:, 0])
    #
    gt_qboxes_ymin = (gt_qboxes[:, 1] - df_boxes_cor[:, 1]) / df_boxes[:, 3]
    gt_qboxes_ymin /= np.sqrt(variances[:, 1])
    #
    gt_qboxes_xmax = (gt_qboxes[:, 2] - df_boxes_cor[:, 2]) / df_boxes[:, 2]
    gt_qboxes_xmax /= np.sqrt(variances[:, 2])
    #
    gt_qboxes_ymax = (gt_qboxes[:, 3] - df_boxes_cor[:, 3]) / df_boxes[:, 3]
    gt_qboxes_ymax /= np.sqrt(variances[:, 3])
    #
    gt_qboxes_x1 = (gt_qboxes[:, 4] -
                    df_boxes_vertices[:, 0, 0]) / df_boxes[:, 2]
    gt_qboxes_x1 /= np.sqrt(variances[:, 4])
    #
    gt_qboxes_y1 = (gt_qboxes[:, 5] -
                    df_boxes_vertices[:, 0, 1]) / df_boxes[:, 3]
    gt_qboxes_y1 /= np.sqrt(variances[:, 5])
    #
    gt_qboxes_x2 = (gt_qboxes[:, 6] -
                    df_boxes_vertices[:, 1, 0]) / df_boxes[:, 2]
    gt_qboxes_x2 /= np.sqrt(variances[:, 6])
    #
    gt_qboxes_y2 = (gt_qboxes[:, 7] -
                    df_boxes_vertices[:, 1, 1]) / df_boxes[:, 3]
    gt_qboxes_y2 /= np.sqrt(variances[:, 7])
    #
    gt_qboxes_x3 = (gt_qboxes[:, 8] -
                    df_boxes_vertices[:, 2, 0]) / df_boxes[:, 2]
    gt_qboxes_x3 /= np.sqrt(variances[:, 8])
    #
    gt_qboxes_y3 = (gt_qboxes[:, 9] -
                    df_boxes_vertices[:, 2, 1]) / df_boxes[:, 3]
    gt_qboxes_y3 /= np.sqrt(variances[:, 9])
    #
    gt_qboxes_x4 = (gt_qboxes[:, 10] -
                    df_boxes_vertices[:, 3, 0]) / df_boxes[:, 2]
    gt_qboxes_x4 /= np.sqrt(variances[:, 10])
    #
    gt_qboxes_y4 = (gt_qboxes[:, 11] -
                    df_boxes_vertices[:, 3, 1]) / df_boxes[:, 3]
    gt_qboxes_y4 /= np.sqrt(variances[:, 11])
    y[:, -20] = gt_qboxes_xmin
    y[:, -19] = gt_qboxes_ymin
    y[:, -18] = gt_qboxes_xmax
    y[:, -17] = gt_qboxes_ymax
    y[:, -16] = gt_qboxes_x1
    y[:, -15] = gt_qboxes_y1
    y[:, -14] = gt_qboxes_x2
    y[:, -13] = gt_qboxes_y2
    y[:, -12] = gt_qboxes_x3
    y[:, -11] = gt_qboxes_y3
    y[:, -10] = gt_qboxes_x4
    y[:, -9] = gt_qboxes_y4
    return y
