import cv2
import random
import numpy as np


def random_hue(
    image,
    label=None,
    min_delta=-18,
    max_delta=18
):
    """ Changes the Hue of an image by adding/subtracting a delta value
    to/from each value in the Hue channel of the image. The image format
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
        - min_delta is less than -360.0
        - max_delta is larger than 360.0

    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/

    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert min_delta >= -360.0, "min_delta must be larger than -360.0"
    assert max_delta <= 360.0, "max_delta must be less than 360.0"

    temp_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    temp_image = np.array(temp_image, dtype=np.float)
    d = random.uniform(min_delta, max_delta)
    temp_image[:, :, 0] += d
    temp_image = np.clip(temp_image, 0, 360)
    temp_image = np.uint8(temp_image)
    temp_image = cv2.cvtColor(temp_image, cv2.COLOR_HSV2BGR)
    return temp_image, label
