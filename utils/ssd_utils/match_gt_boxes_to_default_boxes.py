import numpy as np
from utils.bbox_utils import iou, center_to_corner


def match_gt_boxes_to_default_boxes(
    gt_boxes,
    default_boxes,
    match_threshold=0.5,
    neutral_threshold=0.3
):
    """ Matches ground truth bounding boxes to default boxes based on the SSD paper.

    'We begin by matching each ground truth box to the default box with the best jaccard overlap (as in MultiBox [7]).
    Unlike MultiBox, we then match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)'

    Args:
        - gt_boxes: A numpy array or tensor of shape (num_gt_boxes, 4). Structure [cx, cy, w, h]
        - default_boxes: A numpy array of tensor of shape (num_default_boxes, 4). Structure [cx, cy, w, h]
        - threshold: A float representing a target to decide whether the box is matched
        - default_boxes: A numpy array of tensor of shape (num_default_boxes, 4). Structure [cx, cy, w, h]

    Returns:
        - matches: A numpy array of shape (num_matches, 2). The first index in the last dimension is the index
          of the ground truth box and the last index is the default box index.
        - neutral_boxes: A numpy array of shape (num_neutral_boxes, 2). The first index in the last dimension is the index
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

    # convert gt_boxes and default_boxes to [xmin, ymin, xmax, ymax]
    gt_boxes = center_to_corner(gt_boxes)
    default_boxes = center_to_corner(default_boxes)

    num_gt_boxes = gt_boxes.shape[0]
    num_default_boxes = default_boxes.shape[0]

    matches = np.zeros((num_gt_boxes, 2), dtype=np.int)

    # match ground truth to default box with highest iou
    for i in range(num_gt_boxes):
        gt_box = gt_boxes[i]
        gt_box = np.tile(
            np.expand_dims(gt_box, axis=0),
            (num_default_boxes, 1)
        )
        ious = iou(gt_box, default_boxes)
        matches[i] = [i, np.argmax(ious)]

    # match default boxes to ground truths with overlap higher than threshold
    gt_boxes = np.tile(np.expand_dims(gt_boxes, axis=1), (1, num_default_boxes, 1))
    default_boxes = np.tile(np.expand_dims(default_boxes, axis=0), (num_gt_boxes, 1, 1))
    ious = iou(gt_boxes, default_boxes)
    ious[:, matches[:, 1]] = 0

    matched_gt_boxes_idxs = np.argmax(ious, axis=0)  # for each default boxes, select the ground truth box that has the highest iou
    matched_ious = ious[matched_gt_boxes_idxs, list(range(num_default_boxes))]  # get iou scores between gt and default box that were selected above
    matched_df_boxes_idxs = np.nonzero(matched_ious >= match_threshold)[0]  # select only matched default boxes that has iou larger than threshold
    matched_gt_boxes_idxs = matched_gt_boxes_idxs[matched_df_boxes_idxs]

    # concat the results of the two matching process together
    matches = np.concatenate([
        matches,
        np.concatenate([
            np.expand_dims(matched_gt_boxes_idxs, axis=-1),
            np.expand_dims(matched_df_boxes_idxs, axis=-1)
        ], axis=-1),
    ], axis=0)
    ious[:, matches[:, 1]] = 0

    # find neutral boxes (ious that are higher than neutral_threshold but below threshold)
    # these boxes are neither background nor has enough ious score to qualify as a match.
    background_gt_boxes_idxs = np.argmax(ious, axis=0)
    background_gt_boxes_ious = ious[background_gt_boxes_idxs, list(range(num_default_boxes))]
    neutral_df_boxes_idxs = np.nonzero(background_gt_boxes_ious >= neutral_threshold)[0]
    neutral_gt_boxes_idxs = background_gt_boxes_idxs[neutral_df_boxes_idxs]
    neutral_boxes = np.concatenate([
        np.expand_dims(neutral_gt_boxes_idxs, axis=-1),
        np.expand_dims(neutral_df_boxes_idxs, axis=-1)
    ], axis=-1)

    return matches, neutral_boxes
