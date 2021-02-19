import cv2
import random
import numpy as np


def random_brightness(
    image,
    bboxes=None,
    classes=None,
    min_delta=-32,
    max_delta=32,
    p=0.5
):
    """ Changes the brightness of an image by adding/subtracting a delta value to/from each pixel.
    The image format is assumed to be BGR to match Opencv's standard.

    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes.
        - classes: the list of classes associating with each bounding boxes.
        - min_delta: minimum delta value.
        - max_delta: maximum delta value.
        - p: The probability with which the brightness is changed

    Returns:
        - image: The modified image
        - bboxes: The unmodified bounding boxes
        - classes: The unmodified bounding boxes

    Raises:
        - min_delta is less than -255.0
        - max_delta is larger than 255.0
        - p is smaller than zero
        - p is larger than 1

    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/

    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert min_delta >= -255.0, "min_delta must be larger than -255.0"
    assert max_delta <= 255.0, "max_delta must be less than 255.0"
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    if (random.random() > p):
        return image, bboxes, classes

    temp_image = image.copy()
    d = random.uniform(min_delta, max_delta)
    temp_image += d
    temp_image = np.clip(temp_image, 0, 255)
    return temp_image, bboxes, classes
