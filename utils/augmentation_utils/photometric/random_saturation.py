import cv2
import random
import numpy as np


def random_saturation(
    image,
    label=None,
    min_delta=0.5,
    max_delta=1.5,
    p=0.5
):
    """ Changes the saturation of an image by increasing/decreasing each
    value in the saturation channel by a factor of delta. The image format
    is assumed to be BGR to match Opencv's standard.

    Args:
        - image: the input image.
        - label: the label associate with the objects in the image.
        - min_delta: minimum delta value.
        - max_delta: maximum delta value.

    Returns:
        - image: The modified image
        - label: The unmodified label

    Raises:
        - min_delta is less than 0
        - max_delta is less than min_delta
        - p is smaller than zero
        - p is larger than 1

    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/

    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    """
    assert min_delta >= 0.0, "min_delta must be larger than zero"
    assert max_delta >= min_delta, "max_delta must be larger than min_delta"
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    if (random.random() > p):
        return image, label

    temp_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    temp_image = np.array(temp_image, dtype=np.float)
    d = random.uniform(min_delta, max_delta)
    temp_image[:, :, 1] *= d
    temp_image = np.uint8(temp_image)
    temp_image = cv2.cvtColor(temp_image, cv2.COLOR_HSV2BGR)
    return temp_image, label
