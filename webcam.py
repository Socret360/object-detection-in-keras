import cv2
import numpy as np
from utils import textboxes_utils, augmentation_utils

image, quads = textboxes_utils.read_sample(
    image_path="output/icdar-2015/train/images/img_1.jpg",
    label_path="output/icdar-2015/train/labels/gt_img_1.txt"
)

print(quads)

image, quads, classes = augmentation_utils.random_crop_quad(
    image,
    quads,
    ["text" for i in quads]
)

cv2.imshow("original_image", np.uint8(image))

if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
