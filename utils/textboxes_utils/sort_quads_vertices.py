import numpy as np
from .get_bboxes_from_quads import get_bboxes_from_quads
from utils import bbox_utils
import math


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
    bboxes = np.tile(bboxes, (1, 4, 1))
    #
    temp_quads = quads.copy()
    temp_quads = np.tile(temp_quads, (1, 1, 4))
    temp_quads = np.reshape(temp_quads, (num_quads, 16, 2))
    #
    distances = np.sqrt(np.power(bboxes[:, :, 0] - temp_quads[:, :, 0], 2) + np.power(bboxes[:, :, 1] - temp_quads[:, :, 1], 2))
    distances = np.reshape(distances, (num_quads, 4, 4))
    distances = np.sum(distances, axis=-1)
    #
    delta_ms = np.argmax(distances, axis=-1)
    delta_ms = np.expand_dims(delta_ms, axis=-1)
    delta_ms = np.tile(delta_ms, (1, 4))
    #
    i = np.tile(np.expand_dims(np.array([1, 2, 3, 4]), axis=0), (num_quads, 1))
    i_primes = (i + delta_ms - 1) % 4 + 1
    #
    for idx in range(num_quads):
        # print("=====")
        # print(f"i_primes[idx] - 1: {i_primes[idx] - 1}")
        # print(f"i[idx] - 1: {i[idx] - 1}")
        # print(f"quads_prime[idx, i[idx] - 1]: {quads_prime[idx, i[idx] - 1]}")
        # print(f"quads[idx, i_primes[idx] - 1]: {quads[idx, i_primes[idx] - 1]}")
        # print("=====")
        quads_prime[idx, i[idx] - 1] = quads[idx, i_primes[idx] - 1]

    return quads_prime
