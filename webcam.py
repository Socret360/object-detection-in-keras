import os
import cv2
import json
import argparse
import numpy as np
from networks import SSD_VGG16
from tensorflow.keras.applications import vgg16, mobilenet_v2
from utils import bbox_utils
from networks import SSD_MOBILENETV2

parser = argparse.ArgumentParser(
    description='run inference from images on webcam.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('weights', type=str, help='path to the weight file.')
parser.add_argument('--label_maps', type=str, help='path to label maps file.')
parser.add_argument('--confidence_threshold', type=float,
                    help='the confidence score a detection should match in order to be counted.', default=0.9)
parser.add_argument('--num_predictions', type=int,
                    help='the number of detections to be output as final detections', default=10)
args = parser.parse_args()

with open(args.config, "r") as config_file:
    config = json.load(config_file)

input_size = config["model"]["input_size"]
model_config = config["model"]

if model_config["name"] == "ssd_mobilenetv2":
    with open(args.label_maps, "r") as file:
        label_maps = [line.strip("\n") for line in file.readlines()]
    model = SSD_MOBILENETV2(
        config,
        label_maps,
        is_training=False,
        num_predictions=args.num_predictions)
    process_input_fn = mobilenet_v2.preprocess_input
else:
    print("model have not been implemented")
    exit()

model.load_weights(args.weights)

webcam = cv2.VideoCapture(0)

while True:
    check, image = webcam.read()
    display_image = image.copy()
    image_height, image_width, _ = image.shape
    height_scale, width_scale = input_size/image_height, input_size/image_width

    image = cv2.resize(image, (input_size, input_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = process_input_fn(image)

    image = np.expand_dims(image, axis=0)
    y_pred = model.predict(image)

    for i, pred in enumerate(y_pred[0]):
        classname = label_maps[int(pred[0]) - 1].upper()
        confidence_score = pred[1]

        score = f"{'%.2f' % (confidence_score * 100)}%"
        print(f"-- {classname}: {score}")

        if confidence_score <= 1 and confidence_score > args.confidence_threshold:
            xmin = max(int(pred[2] / width_scale), 1)
            ymin = max(int(pred[3] / height_scale), 1)
            xmax = min(int(pred[4] / width_scale), image_width-1)
            ymax = min(int(pred[5] / height_scale), image_height-1)
            x1 = max(min(int(pred[6] / width_scale), image_width), 0)
            y1 = max(min(int(pred[7] / height_scale), image_height), 0)
            x2 = max(min(int(pred[8] / width_scale), image_width), 0)
            y2 = max(min(int(pred[9] / height_scale), image_height), 0)
            x3 = max(min(int(pred[10] / width_scale), image_width), 0)
            y3 = max(min(int(pred[11] / height_scale), image_height), 0)
            x4 = max(min(int(pred[12] / width_scale), image_width), 0)
            y4 = max(min(int(pred[13] / height_scale), image_height), 0)

            quad = np.array(
                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int)

            cv2.putText(
                display_image,
                classname,
                (int(xmin), int(ymin)),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (100, 100, 255),
                1, 1)

            cv2.polylines(
                display_image,
                [quad],
                True,
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                display_image,
                (xmin, ymin),
                (xmax, ymax),
                (255, 0, 0),
                1
            )

    cv2.imshow('video', display_image)

    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
