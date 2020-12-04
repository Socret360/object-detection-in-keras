import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from utils import get_number_default_boxes


class DefaultBoxes(Layer):
    """A custom layer that generates default boxes for a given feature map. The layer takes a feature of size N and output a tensor of size (batch_size, N, N, 8).
    """

    def __init__(self,
                 image_shape,
                 scale,
                 next_scale,
                 aspect_ratios,
                 extra_default_for_ar_1=True,
                 normalize_coords=True,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 offset=(0.5, 0.5),
                 **kwargs):
        self.image_shape = image_shape
        self.scale = scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.extra_default_for_ar_1 = extra_default_for_ar_1
        self.normalize_coords = normalize_coords
        self.variances = variances
        self.offset = offset
        super(DefaultBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        _, feature_map_height, feature_map_width, feature_map_channels = input_shape
        self.feature_map_height = feature_map_height
        self.feature_map_width = feature_map_width
        self.feature_map_channels = feature_map_channels
        super(DefaultBoxes, self).build(input_shape)

    def call(self, inputs):
        image_height, image_width, _ = self.image_shape
        image_size = min(image_height, image_width)
        feature_map_size = min(self.feature_map_height, self.feature_map_width)
        grid_size = image_size / feature_map_size
        offset_x, offset_y = self.offset
        num_default_boxes = get_number_default_boxes(
            self.aspect_ratios,
            extra_box_for_ar_1=self.extra_default_for_ar_1
        )
        # get all width and height of default boxes
        wh_list = []
        for ar in self.aspect_ratios:
            if ar == 1.0 and self.extra_default_for_ar_1:
                wh_list.append([
                    image_size * np.sqrt(self.scale * self.next_scale) * np.sqrt(ar),
                    image_size * np.sqrt(self.scale * self.next_scale) * (1 / np.sqrt(ar)),
                ])
            wh_list.append([
                image_size * self.scale * np.sqrt(ar),
                image_size * self.scale * (1 / np.sqrt(ar)),
            ])
        wh_list = np.array(wh_list, dtype=np.float)
        # get all center points of each grid cells
        cx = np.linspace(offset_x * grid_size, image_size - (offset_x * grid_size), feature_map_size)
        cy = np.linspace(offset_y * grid_size, image_size - (offset_y * grid_size), feature_map_size)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid, cy_grid = np.expand_dims(cx_grid, axis=-1), np.expand_dims(cy_grid, axis=-1)
        cx_grid, cy_grid = np.tile(cx_grid, (1, 1, num_default_boxes)), np.tile(cy_grid, (1, 1, num_default_boxes))
        #
        default_boxes = np.zeros((self.feature_map_height, self.feature_map_width, num_default_boxes, 4))
        default_boxes[:, :, :, 0] = cx_grid
        default_boxes[:, :, :, 1] = cy_grid
        default_boxes[:, :, :, 2] = wh_list[:, 0]
        default_boxes[:, :, :, 3] = wh_list[:, 1]
        if self.normalize_coords:
            default_boxes[:, :, :, [0, 2]] /= image_size
            default_boxes[:, :, :, [1, 3]] /= image_size
        variances_tensor = np.zeros_like(default_boxes)
        variances_tensor += self.variances
        default_boxes = np.concatenate([default_boxes, variances_tensor], axis=-1)
        default_boxes = np.expand_dims(default_boxes, axis=0)
        default_boxes = K.constant(default_boxes, dtype='float32')
        default_boxes = K.tile(default_boxes, (K.shape(inputs)[0], 1, 1, 1, 1))
        return default_boxes

    def get_config(self):
        config = {
            "image_shape": self.image_shape,
            "scale": self.scale,
            "next_scale": self.next_scale,
            "aspect_ratios": self.aspect_ratios,
            "extra_default_for_ar_1": self.extra_default_for_ar_1,
            "normalize_coords": self.normalize_coords,
            "variances": self.variances,
            "offset": self.offset,
            "feature_map_height": self.feature_map_height,
            "feature_map_width": self.feature_map_width,
            "feature_map_channels": self.feature_map_channels,
        }
        base_config = super(DefaultBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
