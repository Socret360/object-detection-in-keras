import numpy as np
from .get_number_default_quads import get_number_default_quads


def generate_default_quads_for_feature_map(
    feature_map_size,
    image_size,
    offset,
    scale,
    next_scale,
    aspect_ratios,
    angles,
    variances,
    extra_box_for_ar_1
):
    assert len(offset) == 2, "offset must be of len 2"

    grid_size = image_size / feature_map_size
    offset_x, offset_y = offset
    num_default_quads = get_number_default_quads(
        aspect_ratios,
        angles,
        extra_box_for_ar_1=extra_box_for_ar_1
    )
    # get all width and height of default boxes
    wh_list = []
    for angle in angles:
        for ar in aspect_ratios:
            if ar == 1.0 and extra_box_for_ar_1:
                wh_list.append([
                    image_size * np.sqrt(scale * next_scale) * np.sqrt(ar),
                    image_size * np.sqrt(scale * next_scale) * (1 / np.sqrt(ar)),
                    np.deg2rad(angle),
                ])
            wh_list.append([
                image_size * scale * np.sqrt(ar),
                image_size * scale * (1 / np.sqrt(ar)),
                np.deg2rad(angle),
            ])

    wh_list = np.array(wh_list, dtype=np.float)
    cx = np.linspace(offset_x * grid_size, image_size - (offset_x * grid_size), feature_map_size)
    cy = np.linspace(offset_y * grid_size, image_size - (offset_y * grid_size), feature_map_size)
    cx_grid, cy_grid = np.meshgrid(cx, cy)
    cx_grid, cy_grid = np.expand_dims(cx_grid, axis=-1), np.expand_dims(cy_grid, axis=-1)
    cx_grid, cy_grid = np.tile(cx_grid, (1, 1, num_default_quads)), np.tile(cy_grid, (1, 1, num_default_quads))

    x1 = cx_grid - (wh_list[:, 0] / 2)
    y1 = cy_grid - (wh_list[:, 1] / 2)
    x2 = cx_grid + (wh_list[:, 0] / 2)
    y2 = cy_grid - (wh_list[:, 1] / 2)
    x3 = cx_grid + (wh_list[:, 0] / 2)
    y3 = cy_grid + (wh_list[:, 1] / 2)
    x4 = cx_grid - (wh_list[:, 0] / 2)
    y4 = cy_grid + (wh_list[:, 1] / 2)

    rx1 = cx_grid + (x1 - cx_grid) * np.cos(wh_list[:, 2]) + (y1 - cy_grid)*np.sin(wh_list[:, 2])
    ry1 = cy_grid - (x1 - cx_grid) * np.sin(wh_list[:, 2]) + (y1 - cy_grid)*np.cos(wh_list[:, 2])
    rx2 = cx_grid + (x2 - cx_grid) * np.cos(wh_list[:, 2]) + (y2 - cy_grid)*np.sin(wh_list[:, 2])
    ry2 = cy_grid - (x2 - cx_grid) * np.sin(wh_list[:, 2]) + (y2 - cy_grid)*np.cos(wh_list[:, 2])
    rx3 = cx_grid + (x3 - cx_grid) * np.cos(wh_list[:, 2]) + (y3 - cy_grid)*np.sin(wh_list[:, 2])
    ry3 = cy_grid - (x3 - cx_grid) * np.sin(wh_list[:, 2]) + (y3 - cy_grid)*np.cos(wh_list[:, 2])
    rx4 = cx_grid + (x4 - cx_grid) * np.cos(wh_list[:, 2]) + (y4 - cy_grid)*np.sin(wh_list[:, 2])
    ry4 = cy_grid - (x4 - cx_grid) * np.sin(wh_list[:, 2]) + (y4 - cy_grid)*np.cos(wh_list[:, 2])

    default_quads = np.zeros((feature_map_size, feature_map_size, num_default_quads, 8))
    default_quads[:, :, :, 0] = rx1
    default_quads[:, :, :, 1] = ry1
    default_quads[:, :, :, 2] = rx2
    default_quads[:, :, :, 3] = ry2
    default_quads[:, :, :, 4] = rx3
    default_quads[:, :, :, 5] = ry3
    default_quads[:, :, :, 6] = rx4
    default_quads[:, :, :, 7] = ry4
    default_quads[:, :, :, [0, 2, 4, 6]] /= image_size
    default_quads[:, :, :, [1, 3, 5, 7]] /= image_size

    variances_tensor = np.zeros_like(default_quads)
    variances_tensor += variances
    default_quads = np.concatenate([default_quads, variances_tensor], axis=-1)
    return default_quads
