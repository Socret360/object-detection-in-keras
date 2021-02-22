import numpy as np


def center_to_vertices(boxes):
    """ Convert bounding boxes from center format (cx, cy, width, height) to vertices format (x1, y1, x2, y2, x3, y3, x4, y4)
    where (x1, y1) is the top left vertice.

    Args:
        - boxes: numpy array of tensor containing all the boxes to be converted

    Returns:
        - A numpy array of shape (n, 4, 2)
    """
    temp = np.zeros((boxes.shape[0], 8))
    half_width = boxes[..., 2] / 2
    half_height = boxes[..., 3] / 2
    temp[..., 0] = boxes[..., 0] - half_width
    temp[..., 1] = boxes[..., 1] - half_height
    temp[..., 2] = boxes[..., 0] + half_width
    temp[..., 3] = boxes[..., 1] - half_height
    temp[..., 4] = boxes[..., 0] + half_width
    temp[..., 5] = boxes[..., 1] + half_height
    temp[..., 6] = boxes[..., 0] - half_width
    temp[..., 7] = boxes[..., 1] + half_height
    return np.reshape(temp, (temp.shape[0], 4, 2))
