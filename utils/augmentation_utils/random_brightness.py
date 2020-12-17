import cv2
import random
import numpy as np


def random_brightness(
    image,
    label=None,
    min_delta=-32,
    max_delta=32
):
    """ Changes the brightness of an image by adding/subtracting a delta value to/from each pixel.
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
        - min_delta is less than -255.0
        - max_delta is larger than 255.0

    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/
    """
    assert min_delta >= -255.0, "min_delta must be larger than -255.0"
    assert max_delta <= 255.0, "max_delta must be less than 255.0"

    temp_image = np.array(image, dtype=np.float)
    d = random.uniform(min_delta, max_delta)
    temp_image += d
    temp_image = np.clip(temp_image, 0, 255)
    temp_image = np.uint8(temp_image)
    return temp_image, label
