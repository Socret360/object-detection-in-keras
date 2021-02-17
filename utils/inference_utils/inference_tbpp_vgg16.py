from networks import TBPP_VGG16
from tensorflow.keras.applications import vgg16


def inference_tbpp_vgg16(config, args):
    """"""
    model = TBPP_VGG16(
        config,
        is_training=False,
        num_predictions=args.num_predictions)
    process_input_fn = vgg16.preprocess_input
    return model, ["text"], process_input_fn
