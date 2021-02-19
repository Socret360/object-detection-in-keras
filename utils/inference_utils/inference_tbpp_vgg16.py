import cv2
import numpy as np
from networks import TBPP_VGG16
from tensorflow.keras.applications import vgg16
from utils import textboxes_utils


def inference_tbpp_vgg16(config, args):
    """"""
    model = TBPP_VGG16(
        config,
        is_training=False,
        num_predictions=args.num_predictions)
    process_input_fn = vgg16.preprocess_input

    image, quads = textboxes_utils.read_sample(
        image_path=args.input_image,
        label_path=args.label_file
    )

    return model, ["text"], process_input_fn, np.uint8(image), quads, ["text" for i in range(quads.shape[0])]
