import numpy as np
from .get_bboxes_from_quads import get_bboxes_from_quads
from utils import bbox_utils
import math
import cv2


def sort_quads_vertices(quads_prime):
    """ Sort quadrilateral vertices.

    Args:
        - quads_prime: A numpy of shape (n, 4, 2) representing the quadrilaterals.

    Returns:
        - A numpy array with the same shape as quads but its boxes are sorted based on the logic from Liao, Shi & Bai (2018).

    Paper References:
        - Liao, M., Shi, B., & Bai, X. (2018). TextBoxes++: A Single-Shot Oriented Scene Text Detector. https://arxiv.org/abs/1512.02325
    """
    num_quads = quads_prime.shape[0]
    quads = quads_prime.copy()
    bboxes = get_bboxes_from_quads(quads_prime)
    bboxes = bbox_utils.center_to_vertices(bboxes)

    deltas = np.reshape(np.tile(np.reshape(np.expand_dims(np.array([0, 1, 2, 3]), axis=0), (4, 1)), (1, 4)), (16, 1))
    i = np.reshape(np.tile(np.expand_dims(np.array([1, 2, 3, 4]), axis=0), (1, 4)), (16, 1))
    q_indexes = (i + deltas - 1) % 4 + 1
    indexes = np.concatenate([i, q_indexes], axis=-1)

    pts_b = bboxes[:, indexes[:, 0] - 1]
    pts_q = quads[:, indexes[:, 1] - 1]
    distance = np.sqrt((pts_b[..., 0] - pts_q[..., 0]) ** 2 + (pts_b[..., 1] - pts_q[..., 1]) ** 2)
    distance = np.reshape(distance, (num_quads, 4, 4))
    distance = np.sum(distance, axis=-1)

    delta_ms = np.argmin(distance, axis=-1)
    delta_ms = np.expand_dims(delta_ms, axis=-1)
    delta_ms = np.tile(delta_ms, (1, 4))
    delta_ms = np.reshape(delta_ms, (num_quads, 4, 1))

    i_prime = np.array([1, 2, 3, 4])
    i_prime = np.expand_dims(i_prime, axis=-1)
    i_prime = np.expand_dims(i_prime, axis=0)
    i_prime = np.tile(i_prime, (num_quads, 1, 1))
    q_idx_prime = (i_prime + delta_ms - 1) % 4 + 1
    i_prime = np.reshape(i_prime, (num_quads, 4)) - 1
    q_idx_prime = np.reshape(q_idx_prime, (num_quads, 4)) - 1

    for i in range(num_quads):
        quads[i, i_prime[i]] = quads_prime[i, q_idx_prime[i]]

    return quads
