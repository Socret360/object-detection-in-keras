import cv2
import random
import numpy as np
from utils.bbox_utils import iou


def random_crop(
    image,
    bboxes,
    classes,
    min_size=0.1,
    max_size=1,
    min_ar=1,
    max_ar=2,
    overlap_modes=[
        None,
        [0.1, None],
        [0.3, None],
        [0.7, None],
        [0.9, None],
        [None, None],
    ],
    max_attempts=10,
    p=0.5
):
    """ Randomly crops a patch from the image.

    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes.
        - classes: the list of classes associating with each bounding boxes.
        - min_size: the maximum size a crop can be
        - max_size: the maximum size a crop can be
        - min_ar: the minimum aspect ratio a crop can be
        - max_ar: the maximum aspect ratio a crop can be
        - overlap_modes: the list of overlapping modes the function can randomly choose from.
        - max_attempts: the max number of attempts to generate a patch.

    Returns:
        - image: the modified image
        - bboxes: the modified bounding boxes
        - classes: the modified classes

    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/

    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"
    assert min_size > 0, "min_size must be larger than zero."
    assert max_size <= 1, "max_size must be less than or equals to one."
    assert max_size > min_size, "max_size must be larger than min_size."
    assert max_ar > min_ar, "max_ar must be larger than min_ar."
    assert max_attempts > 0, "max_attempts must be larger than zero."

    if (random.random() > p):
        return image, bboxes, classes

    height, width, channels = image.shape
    overlap_mode = random.choice(overlap_modes)

    if overlap_mode == None:
        return image, bboxes, classes

    min_iou, max_iou = overlap_mode

    if min_iou == None:
        min_iou = float(-np.inf)

    if max_iou == None:
        max_iou = float(np.inf)

    temp_image = image.copy()

    for i in range(max_attempts):
        crop_w = random.uniform(min_size * width, max_size * width)
        crop_h = random.uniform(min_size * height, max_size * height)
        crop_ar = crop_h / crop_w

        if crop_ar < min_ar or crop_ar > max_ar:  # crop ar does not match criteria, next attempt
            continue

        crop_left = random.uniform(0, width-crop_w)
        crop_top = random.uniform(0, height-crop_h)

        crop_rect = np.array([crop_left, crop_top, crop_left + crop_w, crop_top + crop_h], dtype=np.float)
        crop_rect = np.expand_dims(crop_rect, axis=0)
        crop_rect = np.tile(crop_rect, (bboxes.shape[0], 1))

        ious = iou(crop_rect, bboxes)

        if ious.min() < min_iou and ious.max() > max_iou:
            continue

        bbox_centers = np.zeros((bboxes.shape[0], 2), dtype=np.float)
        bbox_centers[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
        bbox_centers[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2

        cx_in_crop = (bbox_centers[:, 0] > crop_left) * (bbox_centers[:, 0] < crop_left + crop_w)
        cy_in_crop = (bbox_centers[:, 1] > crop_top) * (bbox_centers[:, 1] < crop_top + crop_h)
        boxes_in_crop = cx_in_crop * cy_in_crop

        if not boxes_in_crop.any():
            continue

        temp_image = temp_image[int(crop_top): int(crop_top+crop_h), int(crop_left): int(crop_left+crop_w), :]
        temp_classes = np.array(classes, dtype=np.object)
        temp_classes = temp_classes[boxes_in_crop]
        temp_bboxes = bboxes[boxes_in_crop]
        crop_rect = np.array([crop_left, crop_top, crop_left + crop_w, crop_top + crop_h], dtype=np.float)
        crop_rect = np.expand_dims(crop_rect, axis=0)
        crop_rect = np.tile(crop_rect, (temp_bboxes.shape[0], 1))
        temp_bboxes[:, :2] = np.maximum(temp_bboxes[:, :2], crop_rect[:, :2])  # if bboxes top left is out of crop then use crop's xmin, ymin
        temp_bboxes[:, :2] -= crop_rect[:, :2]  # translate xmin, ymin to fit crop
        temp_bboxes[:, 2:] = np.minimum(temp_bboxes[:, 2:], crop_rect[:, 2:])
        temp_bboxes[:, 2:] -= crop_rect[:, :2]  # translate xmax, ymax to fit crop
        return temp_image, temp_bboxes, temp_classes.tolist()

    return image, bboxes, classes
