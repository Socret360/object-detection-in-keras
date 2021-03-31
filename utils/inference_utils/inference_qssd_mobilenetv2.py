import cv2
import numpy as np
from networks import QSSD_MOBILENETV2
from tensorflow.keras.applications import mobilenet_v2
from utils import qssd_utils


def inference_qssd_mobilenetv2(config, args):
    with open(args.label_maps, "r") as file:
        label_maps = [line.strip("\n") for line in file.readlines()]

    model = QSSD_MOBILENETV2(
        config,
        label_maps,
        is_training=False,
        num_predictions=args.num_predictions)
    process_input_fn = mobilenet_v2.preprocess_input

    return model, label_maps, process_input_fn
