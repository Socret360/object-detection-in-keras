import numpy as np
from .get_number_default_boxes import get_number_default_boxes
from utils.bbox_utils import center_to_corner, corner_to_center


def generate_default_boxes_for_feature_map(
    feature_map_size,
    image_size,
    offset,
    scale,
    next_scale,
    aspect_ratios,
    variances,
    extra_box_for_ar_1
):
    """ Generates a 4D Tensor representing default boxes.

    Note:
    - The structure of a default box is [xmin, ymin, xmax, ymax]

    Args:
    - feature_map_size: The size of the feature map. (must be square)
    - image_size: The size of the input image. (must be square)
    - offset: The offset for the center of the default boxes. The order is (offset_x, offset_y)
    - scale: The current scale of the default boxes.
    - next_scale: The next scale of the default boxes.
    - aspect_ratios: A list of aspect ratios representing the default boxes.
    - variance: ...
    - extra_box_for_ar_1: Whether to add an extra box for default box with aspect ratio 1.

    Returns:
    - A 4D numpy array of shape (feature_map_size, feature_map_size, num_default_boxes, 8)

    Raises:
    - offset does not have a len of 2

    Code References:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_layers/keras_layer_AnchorBoxes.py
    """
    assert len(offset) == 2, "offset must be of len 2"

    grid_size = image_size / feature_map_size
    offset_x, offset_y = offset
    num_default_boxes = get_number_default_boxes(
        aspect_ratios,
        extra_box_for_ar_1=extra_box_for_ar_1
    )
    # get all width and height of default boxes
    wh_list = []
    for ar in aspect_ratios:
        if ar == 1.0 and extra_box_for_ar_1:
            wh_list.append([
                image_size * np.sqrt(scale * next_scale) * np.sqrt(ar),
                image_size * np.sqrt(scale * next_scale) * (1 / np.sqrt(ar)),
            ])
        wh_list.append([
            image_size * scale * np.sqrt(ar),
            image_size * scale * (1 / np.sqrt(ar)),
        ])
    wh_list = np.array(wh_list, dtype=np.float)
    # get all center points of each grid cells
    cx = np.linspace(offset_x * grid_size, image_size - (offset_x * grid_size), feature_map_size)
    cy = np.linspace(offset_y * grid_size, image_size - (offset_y * grid_size), feature_map_size)
    cx_grid, cy_grid = np.meshgrid(cx, cy)
    cx_grid, cy_grid = np.expand_dims(cx_grid, axis=-1), np.expand_dims(cy_grid, axis=-1)
    cx_grid, cy_grid = np.tile(cx_grid, (1, 1, num_default_boxes)), np.tile(cy_grid, (1, 1, num_default_boxes))
    #
    default_boxes = np.zeros((feature_map_size, feature_map_size, num_default_boxes, 4))
    default_boxes[:, :, :, 0] = cx_grid
    default_boxes[:, :, :, 1] = cy_grid
    default_boxes[:, :, :, 2] = wh_list[:, 0]
    default_boxes[:, :, :, 3] = wh_list[:, 1]
    # clip overflow default boxes
    # default_boxes = center_to_corner(default_boxes)
    # x_coords = default_boxes[:, :, :, [0, 2]]
    # x_coords[x_coords >= image_size] = image_size - 1
    # x_coords[x_coords < 0] = 0
    # default_boxes[:, :, :, [0, 2]] = x_coords
    # y_coords = default_boxes[:, :, :, [1, 3]]
    # y_coords[y_coords >= image_size] = image_size - 1
    # y_coords[y_coords < 0] = 0
    # default_boxes[:, :, :, [1, 3]] = y_coords
    # default_boxes = corner_to_center(default_boxes)
    #
    default_boxes[:, :, :, [0, 2]] /= image_size
    default_boxes[:, :, :, [1, 3]] /= image_size
    #
    variances_tensor = np.zeros_like(default_boxes)
    variances_tensor += variances
    default_boxes = np.concatenate([default_boxes, variances_tensor], axis=-1)
    return default_boxes
