import json
import cv2
import numpy as np
from tensorflow.keras.applications import vgg16
# from networks import TBPP_VGG16, SSD_VGG16
# from data_generators import TBPP_DATA_GENERATOR, SSD_DATA_GENERATOR
from utils import textboxes_utils, augmentation_utils


image, quads = textboxes_utils.read_sample(
    image_path="sample_data/img_1.jpg",
    label_path="sample_data/gt_img_1.txt"
)
augmented_image, augmented_quads, augmented_classes = augmentation_utils.random_expand_quad(
    image=image,
    quads=quads,
    classes=["text" for i in range(quads.shape[0])]
)

image = np.uint8(image)
augmented_image = np.uint8(augmented_image)

for quad in quads:
    cv2.polylines(image, [np.array(quad, dtype=np.int)], True, (0, 255, 0), 1)

for quad in augmented_quads:
    cv2.polylines(augmented_image, [np.array(quad, dtype=np.int)], True, (0, 255, 0), 1)

cv2.imshow('image original', image)
cv2.imshow('image augmented', augmented_image)

if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()

# with open("configs/tbpp300_vgg16.json", "r") as config_file:
#     config = json.load(config_file)

# model = TBPP_VGG16(config)
# model.summary()

# generator = TBPP_DATA_GENERATOR(
#     samples=[
#         "/Users/socretlee/CodingDrive/other/object-detection-in-keras/sample_data/img_29.jpg /Users/socretlee/CodingDrive/other/object-detection-in-keras/sample_data/gt_img_29.txt"
#     ],
#     config=config,
#     shuffle=True,
#     batch_size=1,
#     augment=True,
#     process_input_fn=vgg16.preprocess_input,
# )

# for i, (X, y) in enumerate(generator):
#     if i > 1:
#         break

# image = cv2.imread("sample_data/img_29.jpg")
# with open("sample_data/gt_img_29.txt", "r") as label_file:
#     labels = label_file.readlines()
#     labels = [label.strip("\ufeff").strip("\n") for label in labels]
#     labels = [[int(i) for i in label.split(",")[:-1]] for label in labels]
#     labels = np.array(labels)
#     quads = np.reshape(labels, (labels.shape[0], 4, 2))
#     textboxes_utils.sort_quads_vertices(quads)
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
