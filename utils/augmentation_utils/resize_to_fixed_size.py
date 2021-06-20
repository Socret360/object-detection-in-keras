import cv2
import random
import numpy as np


def resize_to_fixed_size(width, height):
    """ Resize the input image and bounding boxes to fixed size.

    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes.
        - classes: the list of classes associating with each bounding boxes.
        - width: minimum delta value.
        - height: maximum delta value.

    Returns:
        - image: The modified image
        - bboxes: The unmodified bounding boxes
        - classes: The unmodified bounding boxes

    Raises:
        - width is less than 0
        - height is less than 0
    """
    assert width >= 0, "width must be larger than 0"
    assert height >= 0, "height must be larger than 0"

    def _augment(
        image,
        bboxes,
        classes=None
    ):
        temp_image = np.uint8(image)
        o_height, o_width, _ = temp_image.shape
        height_scale, width_scale = height / o_height, width / o_width
        temp_image = cv2.resize(temp_image, (width, height))
        temp_image = np.array(temp_image, dtype=np.float)
        temp_bboxes = bboxes.copy()
        temp_bboxes[:, [0, 2]] *= width_scale
        temp_bboxes[:, [1, 3]] *= height_scale
        temp_bboxes[:, [0, 2]] = np.clip(temp_bboxes[:, [0, 2]], 0, width)
        temp_bboxes[:, [1, 3]] = np.clip(temp_bboxes[:, [1, 3]], 0, height)

        return temp_image, temp_bboxes, classes

    return _augment
