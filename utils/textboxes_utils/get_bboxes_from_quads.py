import numpy as np


def get_bboxes_from_quads(quads):
    """ Extracts minimum bounding rectangle from quadrilaterals.

    Args:
        - quad: A numpy of shape (n, 4, 2) representing the verticies of a quadrilateral.

    Returns:
        - A numpy array with the shape of (n, 4) for cx, cy, width, height
    """
    assert quads.shape[1] == 4 and quads.shape[2] == 2, "quad must have a shape of (n, 4, 2)"
    xmin = np.min(quads[:, :, 0], axis=-1, keepdims=True)
    ymin = np.min(quads[:, :, 1], axis=-1, keepdims=True)
    xmax = np.max(quads[:, :, 0], axis=-1, keepdims=True)
    ymax = np.max(quads[:, :, 1], axis=-1, keepdims=True)
    cx = (xmax + xmin) / 2
    cy = (ymax + ymin) / 2
    width = np.abs(xmax - xmin)
    height = np.abs(ymax - ymin)
    return np.concatenate([cx, cy, width, height], axis=-1)
