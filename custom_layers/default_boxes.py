import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from utils import get_number_default_boxes, generate_default_boxes_for_feature_map


class DefaultBoxes(Layer):
    """A custom layer that generates default boxes for a given feature map.
    Args:
    - image_shape: The shape of the input image
    - scale: The current scale for the default box.
    - next_scale: The next scale for the default box.
    - aspect_ratios: The aspect ratios for the default boxes.
    - offset: The offset for the center of the default boxes.
    - variances: ...
    - extra_box_for_ar_1: Whether to add an extra box for default box with aspect ratio 1.
    - normalize_coords: Whether to normalize the coordinates.
    Returns:
    - A Tensor of shape (batch_size, feature_map_size, feature_map_size, num_default_boxes, 8)
    """

    def __init__(self,
                 image_shape,
                 scale,
                 next_scale,
                 aspect_ratios,
                 variances,
                 offset=(0.5, 0.5),
                 extra_box_for_ar_1=True,
                 normalize_coords=True,
                 **kwargs):
        self.image_shape = image_shape
        self.scale = scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.extra_box_for_ar_1 = extra_box_for_ar_1
        self.normalize_coords = normalize_coords
        self.variances = variances
        self.offset = offset
        super(DefaultBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        _, feature_map_height, feature_map_width, _ = input_shape
        image_height, image_width, _ = self.image_shape

        assert feature_map_height == feature_map_width, "feature map width must be equal to feature map height"
        assert image_height == image_width, "image width must be equal to image height"

        self.feature_map_size = min(feature_map_height, feature_map_width)
        self.image_size = min(image_height, image_width)
        super(DefaultBoxes, self).build(input_shape)

    def call(self, inputs):
        default_boxes = generate_default_boxes_for_feature_map(
            feature_map_size=self.feature_map_size,
            image_size=self.image_size,
            offset=self.offset,
            scale=self.scale,
            next_scale=self.next_scale,
            aspect_ratios=self.aspect_ratios,
            variances=self.variances,
            normalize_coords=self.normalize_coords,
            extra_box_for_ar_1=self.extra_box_for_ar_1
        )
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
            "extra_box_for_ar_1": self.extra_box_for_ar_1,
            "normalize_coords": self.normalize_coords,
            "variances": self.variances,
            "offset": self.offset,
            "feature_map_size": self.feature_map_size,
            "image_size": self.image_size
        }
        base_config = super(DefaultBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
