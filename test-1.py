import cv2
import numpy as np
from utils import ssd_utils, augmentation_utils

image, bboxes, classes = ssd_utils.read_sample(
    "sample_data/justin-aikin-KFJuCzJiQYU-unsplash.jpg",
    "sample_data/justin-aikin-KFJuCzJiQYU-unsplash.xml"
)
augmented_image, augmented_bboxes, augmented_classes = augmentation_utils.random_crop(image, bboxes, classes)
image = np.uint8(image)
augmented_image = np.uint8(augmented_image)

for bbox in bboxes:
    cv2.rectangle(
        image,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        (0, 255, 0),
        10
    )

for bbox in augmented_bboxes:
    cv2.rectangle(
        augmented_image,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        (0, 255, 0),
        10
    )

cv2.imwrite("sample_data/random_crop_original.png", image)
cv2.imwrite("sample_data/random_crop_cropped.png", augmented_image)

