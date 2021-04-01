import os
import cv2
import json
import argparse
import numpy as np
from networks import SSD_VGG16
from tensorflow.keras.applications import vgg16
from utils import bbox_utils


def decode_bboxes(y_pred):
    cx = y_pred[..., -12] * y_pred[..., -4] * y_pred[..., -6] + y_pred[..., -8]
    cy = y_pred[..., -11] * y_pred[..., -3] * y_pred[..., -5] + y_pred[..., -7]
    w = np.exp(y_pred[..., -10] * np.sqrt(y_pred[..., -2])) * y_pred[..., -6]
    h = np.exp(y_pred[..., -9] * np.sqrt(y_pred[..., -1])) * y_pred[..., -5]
    xmin = (cx - 0.5 * w) * input_size
    ymin = (cy - 0.5 * h) * input_size
    xmax = (cx + 0.5 * w) * input_size
    ymax = (cy + 0.5 * h) * input_size
    y_pred = np.concatenate([
        y_pred[..., :-12],
        np.expand_dims(xmin, axis=-1),
        np.expand_dims(ymin, axis=-1),
        np.expand_dims(xmax, axis=-1),
        np.expand_dims(ymax, axis=-1)], -1)
    return y_pred


def confidence_score_threshold_single_class(y_pred, confidence_score_threshold=0.01):
    return y_pred[y_pred[:, 0] > confidence_score_threshold]


def non_max_suppression_single_class(y_pred, confidence_score_threshold=0.45):
    nms_boxes = []
    num_bboxes = y_pred.shape[0]

    for i in range(num_bboxes):
        discard = False
        for j in range(num_bboxes):
            iou = bbox_utils.iou(
                np.expand_dims(y_pred[i], axis=0)[:, -4:],
                np.expand_dims(y_pred[j], axis=0)[:, -4:],
            )

            if iou > confidence_score_threshold:
                score_i = y_pred[i, 0]
                score_j = y_pred[j, 0]
                if (score_j > score_i):
                    discard = True

        if not discard:
            nms_boxes.append(y_pred[i])

    return np.array(nms_boxes)


def top_k(y_pred, k=10):
    sorted_indexes = np.argsort(y_pred[:, 0], axis=-1)
    sorted_indexes = sorted_indexes[::-1]
    return y_pred[sorted_indexes[:k]]


def produce_final(y_pred, confidence_threshold=0.9):
    return y_pred[y_pred[:, 0] >= confidence_threshold]


def draw_bboxes(image, y_pred, show_score=True):
    for pred in y_pred:
        xmin = max(int(pred[-4] / width_scale), 1)
        ymin = max(int(pred[-3] / height_scale), 1)
        xmax = min(int(pred[-2] / width_scale), image_width-1)
        ymax = min(int(pred[-1] / height_scale), image_height-1)
        if show_score:
            cv2.putText(
                image,
                '%.2f' % (pred[0]),
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_PLAIN,
                8,
                (0, 255, 0),
                8, 1)
        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            (255, 0, 0),
            10
        )
    cv2.putText(
        image,
        f"Number of bboxes: {y_pred.shape[0]}",
        (80, 180),
        cv2.FONT_HERSHEY_PLAIN,
        10,
        (0, 255, 0),
        8, 1)


parser = argparse.ArgumentParser(
    description='run inference on an input image.')
parser.add_argument('input_image', type=str, help='path to the input image.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('weights', type=str, help='path to the weight file.')
parser.add_argument('--label_maps', type=str, help='path to label maps file.')
parser.add_argument('--confidence_threshold', type=float,
                    help='the confidence score a detection should match in order to be counted.', default=0.9)
parser.add_argument('--num_predictions', type=int,
                    help='the number of detections to be output as final detections', default=10)
args = parser.parse_args()

assert os.path.exists(args.input_image), "config file does not exist"
assert os.path.exists(args.config), "config file does not exist"
assert args.num_predictions > 0, "num_predictions must be larger than zero"
assert args.confidence_threshold > 0, "confidence_threshold must be larger than zero."
assert args.confidence_threshold <= 1, "confidence_threshold must be smaller than or equal to 1."

with open(args.config, "r") as config_file:
    config = json.load(config_file)

input_size = config["model"]["input_size"]
model_config = config["model"]

with open(args.label_maps, "r") as file:
    label_maps = [line.strip("\n") for line in file.readlines()]

model = SSD_VGG16(
    config,
    label_maps,
    is_training=True,
    num_predictions=10
)

model.load_weights(args.weights, by_name=True)

image = cv2.imread(args.input_image)  # read image in bgr format
image = np.array(image, dtype=np.float)
image = np.uint8(image)

display_image = image.copy()
image_height, image_width, _ = image.shape
height_scale, width_scale = input_size/image_height, input_size/image_width

image = cv2.resize(image, (input_size, input_size))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = vgg16.preprocess_input(image)
image = np.expand_dims(image, axis=0)
y_pred = model.predict(image)[0]
print(f"== all bboxes: {y_pred.shape[0]}")

temp_image = display_image.copy()
y_pred = decode_bboxes(y_pred)
print(f"== decode_bboxes: {y_pred.shape[0]}")
draw_bboxes(temp_image, y_pred, show_score=False)
cv2.imwrite(os.path.join("output", "1-bbox-decode.png"), temp_image)
y_pred = np.concatenate([
    np.expand_dims(y_pred[:, label_maps.index("dog")+1], axis=-1),
    np.expand_dims(y_pred[:, -4], axis=-1),
    np.expand_dims(y_pred[:, -3], axis=-1),
    np.expand_dims(y_pred[:, -2], axis=-1),
    np.expand_dims(y_pred[:, -1], axis=-1),
], axis=-1)


temp_image = display_image.copy()
y_pred = confidence_score_threshold_single_class(y_pred, 0.01)
print(f"== confidence_score_threshold_single_class: {y_pred.shape[0]}")
draw_bboxes(temp_image, y_pred, show_score=False)
cv2.imwrite(os.path.join("output", "2-conf-threshold.png"), temp_image)

temp_image = display_image.copy()
y_pred = non_max_suppression_single_class(y_pred, 0.45)
print(f"== non_max_suppression_single_class: {y_pred.shape[0]}")
draw_bboxes(temp_image, y_pred)
cv2.imwrite(os.path.join("output", "3-nms.png"), temp_image)

temp_image = display_image.copy()
y_pred = top_k(y_pred, k=10)
print(f"== top_k: {y_pred.shape[0]}")
draw_bboxes(temp_image, y_pred)
cv2.imwrite(os.path.join("output", "4-top-k.png"), temp_image)

temp_image = display_image.copy()
y_pred = y_pred[y_pred[:, 0] > 0.9]
print(f"== final results: {y_pred.shape[0]}")
draw_bboxes(temp_image, y_pred)
cv2.imwrite(os.path.join("output", "5-final.png"), temp_image)

# temp_image = display_image.copy()
# y_pred = non_max_suppression(y_pred, label_maps, 0.01)
# select_idxs = np.random.choice(
#     y_pred.shape[0],
#     int(y_pred.shape[0] * 0.01),
#     replace=False
# )
# for pred in y_pred[select_idxs]:
#     xmin = max(int(pred[-4] / width_scale), 1)
#     ymin = max(int(pred[-3] / height_scale), 1)
#     xmax = min(int(pred[-2] / width_scale), image_width-1)
#     ymax = min(int(pred[-1] / height_scale), image_height-1)
#     cv2.rectangle(
#         temp_image,
#         (xmin, ymin),
#         (xmax, ymax),
#         (255, 0, 0),
#         10
#     )
# cv2.imwrite(os.path.join("output", "2-conf-threshold.png"), temp_image)

# print(y_pred.shape)
