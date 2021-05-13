import os
import numpy as np
from networks import SSD_MOBILENET
from tensorflow.keras.applications import mobilenet
from utils import ssd_utils


def ssd_mobilenetv1(config, args):
    assert args.label_maps is not None, "please specify a label map file"
    assert os.path.exists(args.label_maps), "label_maps file does not exist"
    with open(args.label_maps, "r") as file:
        label_maps = [line.strip("\n") for line in file.readlines()]

    model = SSD_MOBILENET(
        config,
        label_maps,
        is_training=False,
        num_predictions=args.num_predictions)
    process_input_fn = mobilenet.preprocess_input
    return model, process_input_fn, label_maps
