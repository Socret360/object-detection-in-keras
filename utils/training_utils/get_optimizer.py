from utils import data_utils
from tensorflow.keras.optimizers import SGD, Adam


def get_optimizer(config, args):
    model_config = config["model"]
    if model_config["name"] == "ssd_vgg16":
        return SGD(
            lr=args.learning_rate,
            momentum=0.9,
            decay=0.0005,
            nesterov=False
        )
    elif model_config["name"] == "ssd_mobilenetv1":
        return SGD(
            lr=args.learning_rate,
            momentum=0.9,
            decay=0.0005,
            nesterov=False
        )
    elif model_config["name"] == "ssd_mobilenetv2":
        return SGD(
            lr=args.learning_rate,
            momentum=0.9,
            decay=0.0005,
            nesterov=False
        )
    elif model_config["name"] == "tbpp_vgg16":
        return Adam(
            lr=args.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=0.001,
            decay=0.0
        )
    else:
        print(f"model with name ${model_config['name']} has not been implemented yet")
        exit()
