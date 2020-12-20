import cv2
import numpy as np
import random


def random_horizontal_flip(
    image,
    bboxes,
    classes,
    p=0.5,
):
    """ Randomly flipped the image horizontally. The image format is assumed to be BGR to match Opencv's standard.

    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes.
        - classes: the list of classes associating with each bounding boxes.
        - p: The probability with which the image is flipped horizontally

    Returns:
        - image: The modified image
        - bboxes: The modified bounding boxes
        - classes: The unmodified bounding boxes

    Raises:
        - p is smaller than zero
        - p is larger than 1

    Webpage References:
        - https://www.kdnuggets.com/2018/09/data-augmentation-bounding-boxes-image-transforms.html/2

    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    if (random.random() > p):
        return image, bboxes, classes

    temp_bboxes = bboxes.copy()
    image_center = np.array(image.shape[:2])[::-1]/2
    image_center = np.hstack((image_center, image_center))
    temp_bboxes[:, [0, 2]] += 2*(image_center[[0, 2]] - temp_bboxes[:, [0, 2]])
    boxes_width = abs(temp_bboxes[:, 0] - temp_bboxes[:, 2])
    temp_bboxes[:, 0] -= boxes_width
    temp_bboxes[:, 2] += boxes_width
    return np.array(cv2.flip(np.uint8(image), 1), dtype=np.float), temp_bboxes, classes
