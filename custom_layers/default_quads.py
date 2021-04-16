import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from utils.qssd_utils import generate_default_quads_for_feature_map


class DefaultQuads(Layer):
    def __init__(
        self,
        image_shape,
        scale,
        next_scale,
        aspect_ratios,
        angles,
        variances,
        offset=(0.5, 0.5),
        extra_box_for_ar_1=True,
        **kwargs
    ):
        self.image_shape = image_shape
        self.scale = scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.angles = angles
        self.extra_box_for_ar_1 = extra_box_for_ar_1
        self.variances = variances
        self.offset = offset
        super(DefaultQuads, self).__init__(**kwargs)

    def build(self, input_shape):
        _, feature_map_height, feature_map_width, _ = input_shape
        image_height, image_width, _ = self.image_shape

        assert feature_map_height == feature_map_width, "feature map width must be equal to feature map height"
        assert image_height == image_width, "image width must be equal to image height"

        self.feature_map_size = min(feature_map_height, feature_map_width)
        self.image_size = min(image_height, image_width)
        super(DefaultQuads, self).build(input_shape)

    def call(self, inputs):
        default_quads = generate_default_quads_for_feature_map(
            feature_map_size=self.feature_map_size,
            image_size=self.image_size,
            offset=self.offset,
            scale=self.scale,
            next_scale=self.next_scale,
            aspect_ratios=self.aspect_ratios,
            angles=self.angles,
            variances=self.variances,
            extra_box_for_ar_1=self.extra_box_for_ar_1
        )
        default_quads = np.expand_dims(default_quads, axis=0)
        default_quads = tf.constant(default_quads, dtype='float32')
        default_quads = tf.tile(default_quads, (tf.shape(inputs)[0], 1, 1, 1, 1))
        return default_quads

    def get_config(self):
        config = {
            "image_shape": self.image_shape,
            "scale": self.scale,
            "next_scale": self.next_scale,
            "aspect_ratios": self.aspect_ratios,
            "angles": self.angles,
            "extra_box_for_ar_1": self.extra_box_for_ar_1,
            "variances": self.variances,
            "offset": self.offset,
            "feature_map_size": self.feature_map_size,
            "image_size": self.image_size
        }
        base_config = super(DefaultQuads, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
