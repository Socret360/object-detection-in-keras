import cv2
import random
import numpy as np


def random_contrast(
    image,
    label=None,
    min_delta=0.5,
    max_delta=1.5
):
    """ Changes the contrast of an image by increasing/decreasing each pixel by a factor of delta.
    The image format is assumed to be BGR to match Opencv's standard.

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

    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/

    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert min_delta >= 0.0, "min_delta must be larger than zero"
    assert max_delta >= min_delta, "max_delta must be larger than min_delta"

    temp_image = np.array(image, dtype=np.float)
    d = random.uniform(min_delta, max_delta)
    temp_image *= d
    temp_image = np.clip(temp_image, 0, 255)
    temp_image = np.uint8(temp_image)
    return temp_image, label
