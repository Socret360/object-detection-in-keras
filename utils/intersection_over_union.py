import numpy as np


def intersection_over_union(box1, box2, eps=10e-5):
    """ Calculates the intersection over union between two boxes also known as Jaccard Index.
    Args:
    - box1:
    - box2:

    Returns:
    -
    """
    assert box1.shape == box2.shape, "The two boxes array must be the same shape."
    xmin_intersect = np.maximum(box1[:, :, 0], box2[:, :, 0])
    ymin_intersect = np.maximum(box1[:, :, 1], box2[:, :, 1])
    xmax_intersect = np.minimum(box1[:, :, 2], box2[:, :, 2])
    ymax_intersect = np.minimum(box1[:, :, 3], box2[:, :, 3])
    intersect = np.abs(xmax_intersect - xmin_intersect) * np.abs(ymax_intersect - ymin_intersect)
    box1_area = np.abs(box1[:, :, 2] - box1[:, :, 0]) * np.abs((box1[:, :, 3] - box1[:, :, 1]))
    box2_area = np.abs(box2[:, :, 2] - box2[:, :, 0]) * np.abs((box2[:, :, 3] - box2[:, :, 1]))
    union = box1_area + box2_area - intersect
    return intersect / (union + eps)
