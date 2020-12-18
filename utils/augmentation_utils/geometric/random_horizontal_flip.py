import cv2
import numpy as np
import random


def random_horizontal_flip(
    image,
    label,
    p=0.5,
):
    """ Randomly flipped the image horizontally. The image format is assumed to be BGR to match Opencv's standard.

    Args:
        - image: the input image.
        - label: the label associate with the objects in the image.
        - p: The probability with which the image is flipped horizontally

    Returns:
        - image: The modified image
        - label: The unmodified label

    Raises:
        - p is smaller than zero
        - p is larger than 1

    Webpage References:
        - https://www.kdnuggets.com/2018/09/data-augmentation-bounding-boxes-image-transforms.html/2
    """
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    if (random.random() > p):
        return image, label

    temp_label = np.array(label, dtype=np.float)
    image_center = np.array(image.shape[:2])[::-1]/2
    image_center = np.hstack((image_center, image_center))
    temp_label[:, [0, 2]] += 2*(image_center[[0, 2]] - temp_label[:, [0, 2]])
    boxes_width = abs(temp_label[:, 0] - temp_label[:, 2])
    temp_label[:, 0] -= boxes_width
    temp_label[:, 2] += boxes_width
    temp_label = temp_label.tolist()
    return cv2.flip(image, 1), temp_label
