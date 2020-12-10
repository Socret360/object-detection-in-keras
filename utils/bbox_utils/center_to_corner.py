import numpy as np


def center_to_corner(boxes):
    """ Convert bounding boxes from center format (cx, cy, width, height) to corner format (xmin, ymin, xmax, ymax)

    Args:
        - boxes: numpy array of tensor containing all the boxes to be converted

    Returns:
        - A numpy array or tensor of converted boxes
    """
    temp = boxes.copy()
    temp[..., 0] = boxes[..., 0] - (boxes[..., 2] / 2)  # xmin
    temp[..., 1] = boxes[..., 1] - (boxes[..., 3] / 2)  # ymin
    temp[..., 2] = boxes[..., 0] + (boxes[..., 2] / 2)  # xmax
    temp[..., 3] = boxes[..., 1] + (boxes[..., 3] / 2)  # ymax
    return temp
