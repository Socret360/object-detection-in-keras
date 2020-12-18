import cv2
import numpy as np


def random_vertical_flip(
    image,
    label,
    p=0.5
):
    """ Randomly flipped the image vertically. The image format is assumed to be BGR to match Opencv's standard.

    Args:
        - image: the input image.
        - label: the label associate with the objects in the image.
        - p: The probability with which the image is flipped vertically

    Returns:
        - image: The modified image
        - label: The unmodified label

    Raises:
        - p is smaller than zero
        - p is larger than 1

    Webpage References:
        - https://www.kdnuggets.com/2018/09/data-augmentation-bounding-boxes-image-transforms.html/2
    """
    temp_label = np.array(label, dtype=np.float)
    image_center = np.array(image.shape[:2])[::-1]/2
    image_center = np.hstack((image_center, image_center))
    temp_label[:, [1, 3]] += 2*(image_center[[1, 3]] - temp_label[:, [1, 3]])
    boxes_height = abs(temp_label[:, 1] - temp_label[:, 3])
    temp_label[:, 1] -= boxes_height
    temp_label[:, 3] += boxes_height
    temp_label = temp_label.tolist()
    return cv2.flip(image, 0), temp_label
