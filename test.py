import json
import cv2
import numpy as np
from networks import TBPP_VGG16
from utils import textboxes_utils

image = cv2.imread("sample_data/img_29.jpg")
with open("sample_data/gt_img_29.txt", "r") as label_file:
    labels = label_file.readlines()
    labels = [label.strip("\ufeff").strip("\n") for label in labels]
    labels = [[int(i) for i in label.split(",")[:-1]] for label in labels]
    labels = np.array(labels)
    quads = np.reshape(labels, (labels.shape[0], 4, 2))
    textboxes_utils.sort_quads_vertices(quads)
    # bboxes = textboxes_utils.get_bboxes_from_quads(quads)
    # bboxes = textboxes_utils.get_bboxes_from_quads(quads)
    # for label in labels:
    # label = np.expand_dims(label, axis=0)
    # quad = np.reshape(label, (-1, 2))
    # cv2.polylines(image, [quad], 1, color=(255, 0, 0), thickness=1)
    # cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)

    # cv2.imshow("image", image)
    # if cv2.waitKey(0) == ord('q'):
    #     cv2.destroyAllWindows()

# with open("configs/tbpp300_vgg16.json", "r") as config_file:
#     config = json.load(config_file)

# model = TBPP_VGG16(config)
# model.summary()
