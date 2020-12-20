import random
import numpy as np


def random_expand(
    image,
    bboxes,
    classes,
    min_ratio=1,
    max_ratio=4,
    mean=[0.406, 0.456, 0.485],  # BGR
    p=0.5
):
    """ Randomly expands an image and bounding boxes by a ratio between min_ratio and max_ratio. The image format is assumed to be BGR to match Opencv's standard.

    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes.
        - classes: the list of classes associating with each bounding boxes.
        - min_ratio: The minimum value to expand the image. Defaults to 1.
        - max_ratio: The maximum value to expand the image. Defaults to 4.
        - p: The probability with which the image is expanded

    Returns:
        - image: The modified image
        - bboxes: The modified bounding boxes
        - classes: The unmodified bounding boxes

    Raises:
        - p is smaller than zero
        - p is larger than 1

    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/

    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"
    assert min_ratio > 0, "min_ratio must be larger than zero"
    assert max_ratio > min_ratio, "max_ratio must be larger than min_ratio"

    if (random.random() > p):
        return image, bboxes, classes

    height, width, depth = image.shape
    ratio = random.uniform(min_ratio, max_ratio)
    left = random.uniform(0, width * ratio - width)
    top = random.uniform(0, height * ratio - height)
    temp_image = np.zeros(
        (int(height * ratio), int(width * ratio), depth),
        dtype=image.dtype
    )
    temp_image[:, :, :] = mean
    temp_image[int(top):int(top+height), int(left):int(left+width)] = image
    temp_bboxes = bboxes.copy()
    temp_bboxes[:, :2] += (int(left), int(top))
    temp_bboxes[:, 2:] += (int(left), int(top))
    return temp_image, temp_bboxes, classes
