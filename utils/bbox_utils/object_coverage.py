import numpy as np


def object_coverage(box_group1, box_group2):
    assert box_group1.shape == box_group2.shape, "The two boxes array must be the same shape."
    xmin_intersect = np.maximum(box_group1[..., 0], box_group2[..., 0])
    ymin_intersect = np.maximum(box_group1[..., 1], box_group2[..., 1])
    xmax_intersect = np.minimum(box_group1[..., 2], box_group2[..., 2])
    ymax_intersect = np.minimum(box_group1[..., 3], box_group2[..., 3])

    intersect = (xmax_intersect - xmin_intersect) * (ymax_intersect - ymin_intersect)
    box_group2_area = (box_group2[..., 2] - box_group2[..., 0]) * (box_group2[..., 3] - box_group2[..., 1])
    res = intersect / box_group2_area
    
    # set invalid ious to zeros
    res[xmax_intersect < xmin_intersect] = 0
    res[ymax_intersect < ymin_intersect] = 0
    res[res < 0] = 0
    res[res > 1] = 0
    return res
