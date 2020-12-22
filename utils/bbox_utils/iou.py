import numpy as np


def iou(box_group1, box_group2):
    """ Calculates the intersection over union (aka. Jaccard Index) between two boxes.
    Boxes are assumed to be in corners format (xmin, ymin, xmax, ymax)

    Args:
    - box_group1: boxes in group 1
    - box_group2: boxes in group 2

    Returns:
    - A numpy array of shape (len(box_group1), len(box_group2)) where each value represents the iou between a box in box_group1 to a box in box_group2

    Raises:
    - The shape of box_group1 and box_group2 are not the same.

    Code References:
    - https://stackoverflow.com/questions/28723670/intersection-over-union-between-two-detections/41660682
    """
    assert box_group1.shape == box_group2.shape, "The two boxes array must be the same shape."
    xmin_intersect = np.maximum(box_group1[..., 0], box_group2[..., 0])
    ymin_intersect = np.maximum(box_group1[..., 1], box_group2[..., 1])
    xmax_intersect = np.minimum(box_group1[..., 2], box_group2[..., 2])
    ymax_intersect = np.minimum(box_group1[..., 3], box_group2[..., 3])

    intersect = (xmax_intersect - xmin_intersect) * (ymax_intersect - ymin_intersect)
    box_group1_area = (box_group1[..., 2] - box_group1[..., 0]) * (box_group1[..., 3] - box_group1[..., 1])
    box_group2_area = (box_group2[..., 2] - box_group2[..., 0]) * (box_group2[..., 3] - box_group2[..., 1])
    union = box_group1_area + box_group2_area - intersect
    res = intersect / union

    # set invalid ious to zeros
    res[xmax_intersect < xmin_intersect] = 0
    res[ymax_intersect < ymin_intersect] = 0
    res[res < 0] = 0
    res[res > 1] = 0
    return res
