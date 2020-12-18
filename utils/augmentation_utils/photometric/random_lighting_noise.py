import cv2
import random
import numpy as np


def random_lighting_noise(
    image,
    label=None,
    p=0.5
):
    """ Changes the lighting of the image by randomly swapping the channels.
    The image format is assumed to be BGR to match Opencv's standard.

    Args:
        - image: the input image.
        - label: the label associate with the objects in the image.
        - p: The probability with which the contrast is changed

    Returns:
        - image: The modified image
        - label: The unmodified label

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

    if (random.random() > p):
        return image, label

    temp_image = np.array(image, dtype=np.float)
    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0)
    ]
    selected_perm = random.randint(0, len(perms) - 1)
    perm = perms[selected_perm]
    temp_image = temp_image[:, :, perm]
    temp_image = np.uint8(temp_image)
    return temp_image, label
