import numpy as np


def match_bounding_boxes_to_default_boxes(bounding_boxes, default_boxes):
    """ Matches ground truth bounding boxes to default boxes based on the SSD paper.
    Note:
    - The structure for a bounding box is [xmin, ymin, xmax, ymax]

    Args:
    - bounding_boxes: A numpy array or tensor of shape (num_bounding_boxes, 4)
    - default_boxes: A numpy array of tensor of shape (num_default_boxes, 4)

    Returns:
    - A numpy array of shape (num_matches, 2). The last shape is 2 for the index of matched bounding box and index of matched default box
    """
    # print(bounding_boxes.shape, default_boxes[:2, :].shape)
    default_boxes = default_boxes[:5, :].copy()
    print(bounding_boxes[:, 0].shape, default_boxes[:, 0].shape)
