import numpy as np
from utils import intersection_over_union


def match_gt_boxes_to_default_boxes(gt_boxes, default_boxes, threshold=0.5):
    """ Matches ground truth bounding boxes to default boxes based on the SSD paper.

    'We begin by matching each ground truth box to the default box with the best jaccard overlap (as in MultiBox [7]).
    Unlike MultiBox, we then match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)'

    Note:
        - The structure for both the ground box and the default box is [xmin, ymin, xmax, ymax]

    Args:
        - gt_boxes: A numpy array or tensor of shape (num_gt_boxes, 4)
        - default_boxes: A numpy array of tensor of shape (num_default_boxes, 4)

    Returns:
        - A numpy array of shape (num_matches, 2). The first index in the last dimension is the index
          of the ground truth box and the last index is the default box index.

    Raises:
        - Either the shape of ground truth's boxes array or the default boxes array is not 2

    Code References:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd_encoder_decoder/matching_utils.py

    Paper References:
        - Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y., & Berg, A. C. (2016).
          SSD: Single Shot MultiBox Detector. https://arxiv.org/abs/1512.02325
    """

    assert len(gt_boxes.shape) == 2, "Shape of ground truth boxes array must be 2"
    assert len(default_boxes.shape) == 2, "Shape of default boxes array must be 2"

    num_gt_boxes = gt_boxes.shape[0]
    num_default_boxes = default_boxes.shape[0]

    matches = np.zeros((num_gt_boxes, 2))

    # match ground truth to default box with highest iou
    for i in range(num_gt_boxes):
        gt_box = gt_boxes[i]
        gt_box = np.tile(np.expand_dims(gt_box, axis=0), (num_default_boxes, 1))
        ious = intersection_over_union(gt_box, default_boxes)
        matches[i] = [i, np.argmax(ious)]

    # match default boxes to ground truths with overlap higher than threshold
    gt_boxes = np.tile(np.expand_dims(gt_boxes, axis=1), (1, num_default_boxes, 1))
    default_boxes = np.tile(np.expand_dims(default_boxes, axis=0), (num_gt_boxes, 1, 1))
    ious = intersection_over_union(gt_boxes, default_boxes)
    matched_gt_boxes_idxs = np.argmax(ious, axis=0)  # for each default boxes, select the ground truth box that has the highest iou
    matched_ious = ious[matched_gt_boxes_idxs, list(range(num_default_boxes))]  # get iou scores between gt and default box that were selected above
    matched_db_boxes_idxs = np.nonzero(matched_ious >= threshold)[0]  # select only matched default boxes that has iou larger than threshold
    matched_gt_boxes_idxs = matched_gt_boxes_idxs[matched_db_boxes_idxs]

    # concat the results of the two matching process together
    matches = np.concatenate([
        matches,
        np.concatenate([
            np.expand_dims(matched_gt_boxes_idxs, axis=-1),
            np.expand_dims(matched_db_boxes_idxs, axis=-1)
        ], axis=-1),
    ], axis=0)

    return matches
