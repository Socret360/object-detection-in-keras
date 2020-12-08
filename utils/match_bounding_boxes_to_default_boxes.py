import numpy as np
from utils import intersection_over_union


def match_bounding_boxes_to_default_boxes(gt_boxes, default_boxes, threshold=0.5):
    """ Matches ground truth bounding boxes to default boxes based on the SSD paper.
    Note:
    - The structure for a bounding box is [xmin, ymin, xmax, ymax]

    Args:
    - bounding_boxes: A numpy array or tensor of shape (num_bounding_boxes, 4)
    - default_boxes: A numpy array of tensor of shape (num_default_boxes, 4)

    Returns:
    - A numpy array of shape (num_matches, 2). The last shape is 2 for the index of matched bounding box and index of matched default box
    """
    gt_boxes = np.tile(np.expand_dims(gt_boxes, axis=1), (1, default_boxes.shape[0], 1))
    default_boxes = np.tile(np.expand_dims(default_boxes, axis=0), (gt_boxes.shape[0], 1, 1))
    iou = intersection_over_union(gt_boxes, default_boxes)
    all_db_indexes = list(range(iou.shape[1]))
    matched_gt = np.argmax(iou, axis=0)  # for each default boxes, select the ground truth box that has the highest iou
    matched_iou = iou[matched_gt, all_db_indexes]  # get iou scores between gt and default box that were selected above
    db_threshold_met = np.nonzero(matched_iou >= threshold)[0]  # select only matched default boxes that has iou larger than threshold
    gt_threshold_met = matched_gt[db_threshold_met]
    return gt_threshold_met, db_threshold_met
