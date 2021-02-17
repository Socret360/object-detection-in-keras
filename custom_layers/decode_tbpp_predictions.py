import tensorflow as tf
from tensorflow.keras.layers import Layer
from utils import textboxes_utils


class DecodeTBPPPredictions(Layer):
    def __init__(
        self,
        input_size,
        nms_max_output_size=400,
        confidence_threshold=0.01,
        iou_threshold=0.45,
        num_predictions=10,
        **kwargs
    ):
        self.input_size = input_size
        self.nms_max_output_size = nms_max_output_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.num_predictions = num_predictions
        super(DecodeTBPPPredictions, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DecodeTBPPPredictions, self).build(input_shape)

    def call(self, inputs):
        y_pred = textboxes_utils.decode_predictions(
            y_pred=inputs,
            input_size=self.input_size,
            nms_max_output_size=self.nms_max_output_size,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            num_predictions=self.num_predictions
        )
        return y_pred

    def get_config(self):
        config = {
            'input_size': self.input_size,
            'nms_max_output_size': self.nms_max_output_size,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'num_predictions': self.num_predictions,
        }
        base_config = super(DecodeTBPPPredictions, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
