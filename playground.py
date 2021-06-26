import os
import cv2
from glob import glob
import numpy as np
from utils import bbox_utils, pascal_voc_utils


samples = list(glob("output/cp_275_loss-3.30_valloss-3.84.h5/*txt"))
dir = "/Users/socretlee/CodingDrive/other/object-detection-in-keras/data/easy_example_tests"
labels_dir = "/Users/socretlee/CodingDrive/other/object-detection-in-keras/data/easy_example_tests/labels"
images_dir = "/Users/socretlee/CodingDrive/other/object-detection-in-keras/data/easy_example_tests/images"

color_dicts = {
    "dog": (60, 76, 231),
    "cat": (113, 204, 46),
}


def draw_text(
    img,
    text,
    font=cv2.FONT_HERSHEY_PLAIN,
    pos=(0, 0),
    font_scale=3,
    font_thickness=2,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0),
):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


grids = []
for idx, classname in enumerate(["dog", "cat"]):
    grid = None

    detections = []

    for sample in samples:
        filename = os.path.basename(sample)
        filename = filename[:filename.index(".")]
        orig_image = cv2.imread(os.path.join(images_dir, f"{filename}.jpg"))
        image = cv2.resize(orig_image, (0, 0), fx=0.3, fy=0.3)
        height_scale, width_scale = image.shape[0] / orig_image.shape[0], image.shape[1] / orig_image.shape[1]
        bboxes, classes = pascal_voc_utils.read_label(os.path.join(labels_dir, f"{filename}.xml"))
        bboxes = np.array(bboxes)
        bboxes[:, [0, 2]] *= width_scale
        bboxes[:, [1, 3]] *= height_scale

        with open(sample, "r") as detections_file:
            _predictions_in_file = [i.strip("\n").split(" ") for i in detections_file.readlines() if i.strip('\n').split(" ")[0] == classname]

        ground_truth_in_file = []
        predictions_in_file = []

        for i, bbox in enumerate(bboxes):
            if classes[i] == classname:
                xmin = max(int(bbox[0]), 10)
                ymin = max(int(bbox[1]), 10)
                xmax = min(int(bbox[2]), image.shape[1]-10)
                ymax = min(int(bbox[3]), image.shape[0]-10)
                ground_truth_in_file.append({
                    "bbox": [xmin, ymin, xmax, ymax],
                    "class": classes[i]
                })
                cv2.rectangle(
                    image,
                    (xmin, ymin),
                    (xmax, ymax),
                    (0, 255, 0),
                    15,
                )

        for i, detection in enumerate(_predictions_in_file):
            bbox = np.array([float(i) for i in detection[2:]])
            bbox[[0, 2]] *= width_scale
            bbox[[1, 3]] *= height_scale
            xmin = max(int(bbox[0]), 10)
            ymin = max(int(bbox[1]), 10)
            xmax = min(int(bbox[2]), image.shape[1]-10)
            ymax = min(int(bbox[3]), image.shape[0]-10)
            predictions_in_file.append({
                "bbox": [xmin, ymin, xmax, ymax],
                "confidence_score": detection[1],
                "class": detection[0],
            })

        gt_pred_matrix = np.zeros((len(ground_truth_in_file), len(predictions_in_file)))

        for pred_i, pred in enumerate(predictions_in_file):
            for gt_i, gt in enumerate(ground_truth_in_file):
                gt_pred_matrix[gt_i][pred_i] = bbox_utils.iou(
                    np.expand_dims(pred["bbox"], axis=0),
                    np.expand_dims(gt["bbox"], axis=0),
                )[0]

        t = np.where(gt_pred_matrix > 0.3, 1, 0)

        for gt_i in range(t.shape[0]):
            row = t[gt_i]
            true_positives_per_gt = np.argwhere(row == 1)
            if len(true_positives_per_gt) > 1:
                t[gt_i] = np.zeros_like(row)
                largest_overlap_idx = np.argmax(row)
                t[gt_i][largest_overlap_idx] = 1

        for pred_idx in range(t.shape[1]):
            cols = t[:, pred_idx]

            if (len(np.argwhere(cols == 1)) > 0):
                detection = {
                    "bbox": predictions_in_file[pred_idx]["bbox"],
                    "confidence_score": float(predictions_in_file[pred_idx]["confidence_score"]),
                    "label": chr(len(detections) + 65),
                    "tp": 1,
                    "fp": 0,
                }
                detections.append(detection)
            else:
                detection = {
                    "bbox": predictions_in_file[pred_idx]["bbox"],
                    "confidence_score": float(predictions_in_file[pred_idx]["confidence_score"]),
                    "label": chr(len(detections) + 65),
                    "tp": 0,
                    "fp": 1,
                }
                detections.append(detection)
            bbox = detection["bbox"]
            xmin = max(int(bbox[0]), 10)
            ymin = max(int(bbox[1]), 10)
            xmax = min(int(bbox[2]), image.shape[1]-10)
            ymax = min(int(bbox[3]), image.shape[0]-10)
            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                (0, 0, 255),
                15,
            )
            draw_text(
                image,
                f"{detection['label']}: {int(float(detection['confidence_score']) * 100)}%",
                pos=(xmin, ymin),
                text_color=(255, 255, 255),
                font_scale=3,
                font_thickness=5,
                text_color_bg=(0, 0, 255)
            )

        if grid is None:
            grid = image
        else:
            image = cv2.resize(image, (image.shape[1], grid.shape[0]))
            grid = cv2.hconcat([grid, image])

    detections.sort(key=lambda x: x["confidence_score"], reverse=True)  # highest confidence score first

    for detection in detections:
        print(f"{detection['label']}: score-{'%.5f' % detection['confidence_score']},  TP-{detection['tp']}, FP-{detection['fp']}")

    cv2.imshow("image", grid)

    if cv2.waitKey(0) == ord('q'):
        cv2.imwrite(f"{idx}.jpg", grid)
        cv2.destroyAllWindows()

    print("\n")
