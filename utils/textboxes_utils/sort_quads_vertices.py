import numpy as np
from .get_bboxes_from_quads import get_bboxes_from_quads
from utils import bbox_utils
import math
import cv2


def sort_quads_vertices(quads):
    """ Sort quadrilateral vertices.

    Args:
        - quads: A numpy of shape (n, 4, 2) representing the quadrilaterals.

    Returns:
        - A numpy array with the same shape as quads but its boxes are sorted based on the logic from Liao, Shi & Bai (2018).

    Paper References:
        - Liao, M., Shi, B., & Bai, X. (2018). TextBoxes++: A Single-Shot Oriented Scene Text Detector. https://arxiv.org/abs/1512.02325
    """
    num_quads = quads.shape[0]
    quads_prime = quads.copy()
    bboxes = get_bboxes_from_quads(quads)
    bboxes = bbox_utils.center_to_vertices(bboxes)

    for idx in range(num_quads):
        quad = quads[idx]
        bbox = bboxes[idx]

        delta_ms = []

        for delta in [0, 1, 2, 3]:
            sums = 0
            for i in [1, 2, 3, 4]:
                q_idx = (i + delta - 1) % 4+1
                pts_b = bbox[i-1]
                pts_q = quad[q_idx-1]
                distance = math.sqrt((pts_b[0] - pts_q[0]) ** 2 + (pts_b[1] - pts_q[1]) ** 2)
                sums += distance
            delta_ms.append(sums)

        delta_m = np.argmin(delta_ms)

        for i in [1, 2, 3, 4]:
            q_idx_prime = (i + delta_m - 1) % 4 + 1
            quads_prime[idx, i - 1] = quads[idx, q_idx_prime - 1]

    return quads_prime
