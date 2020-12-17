import json
import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD
from models.ssd import SSD300_VGG16_ORIGINAL
from losses import SSD_LOSS
from data_generators.voc import SSD_VOC_DATA_GENERATOR
from base_networks import VGG16_D
from utils.augmentation_utils import random_brightness, random_hue, random_saturation, random_contrast

if __name__ == "__main__":
    image = cv2.imread("data/test.jpg")
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    image, label = random_contrast(image)
    cv2.imshow("image", image)

    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
    # base_model = VGG16_D(num_classes=20)
    # with open("configs/ssd300_vgg16_original.json") as config_file:
    #     config = json.load(config_file)

    # model = SSD300_VGG16_ORIGINAL(config)
    # model.summary()
    # model.compile(
    #     optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False),
    #     loss=SSD_LOSS(
    #         alpha=config["training"]["alpha"],
    #         min_negative_boxes=config["training"]["min_negative_boxes"],
    #         negative_boxes_ratio=config["training"]["negative_boxes_ratio"]
    #     ).compute
    # )
    # # data_generator = SSD_VOC_DATA_GENERATOR(
    # #     samples=["data/test.jpg data/test.xml"],
    # #     config=config
    # # )

    # model.fit(
    #     x=SSD_VOC_DATA_GENERATOR(
    #         samples=["data/test.jpg data/test.xml"],
    #         config=config
    #     ),
    #     batch_size=1,
    #     epochs=1,
    #     steps_per_epoch=1,
    # )
