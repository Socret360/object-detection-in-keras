import json
import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD
from models.ssd import SSD300_VGG16_ORIGINAL
from losses import SSD_LOSS
from data_generators.voc import SSD_VOC_DATA_GENERATOR
from base_networks import VGG16_D
import xml.etree.ElementTree as ET
from utils.augmentation_utils import geometric

if __name__ == "__main__":
    image = cv2.imread("data/test.jpg")  # read image in bgr format
    image_height, image_width, _ = image.shape
    xml_root = ET.parse("data/test.xml").getroot()
    objects = xml_root.findall("object")
    bboxes = []
    classes = []
    for i, obj in enumerate(objects):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        bboxes.append([xmin, ymin, xmax, ymax])
        classes.append(name)
    augmented_image, augmented_bboxes = geometric.random_vertical_flip(image=image, label=bboxes, p=1)

    for i, _ in enumerate(bboxes):
        before = bboxes[i]
        after = augmented_bboxes[i]
        cv2.rectangle(
            image,
            (int(before[0]), int(before[1])),
            (int(before[2]), int(before[3])),
            (0, 255 * (i * 0.2), 255 * (i * 0.8)), 2)
        cv2.rectangle(
            augmented_image,
            (int(after[0]), int(after[1])),
            (int(after[2]), int(after[3])),
            (0, 255 * (i * 0.2), 255 * (i * 0.8)), 2)

    cv2.imshow("origin", image)
    cv2.imshow("image", augmented_image)

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
