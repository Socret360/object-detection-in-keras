from glob import glob
import os

from matplotlib import pyplot as plt
from utils import bbox_utils, ssd_utils

import cv2
import numpy as np
from networks.ssd_vgg16 import SSD_VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


def ssd_vgg16(config, args):
    """"""
    input_size = config["model"]["input_size"]

    with open(args.label_maps, "r") as file:
        label_maps = [line.strip("\n") for line in file.readlines()]

    model = SSD_VGG16(
        config,
        label_maps,
        is_training=False,
        num_predictions=args.num_predictions
    )

    model.load_weights(args.weights)

    with open(args.split_file, "r") as split_file:
        samples = [i.strip("\n") for i in split_file.readlines()]

    # samples = samples[:100]

    Xs, ys = np.zeros((len(samples), input_size, input_size, 3), dtype=np.float), []

    for i, sample in enumerate(samples):
        print(f"reading sample: {i+1}/{len(samples)}")
        image_file, label_file = sample.split(" ")
        image, bboxes, classes = ssd_utils.read_sample(
            image_path=os.path.join(args.images_dir, image_file),
            label_path=os.path.join(args.labels_dir, label_file),
        )
        image_height, image_width, _ = image.shape
        height_scale, width_scale = input_size/image_height, input_size/image_width
        input_img = cv2.resize(np.uint8(image), (input_size, input_size))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = preprocess_input(input_img)
        Xs[i] = input_img

        objects = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            xmin = bbox[0] * width_scale
            ymin = bbox[1] * height_scale
            xmax = bbox[2] * width_scale
            ymax = bbox[3] * height_scale
            objects.append({
                "class": classes[i],
                "bbox": [xmin, ymin, xmax, ymax]
            })

        ys.append(objects)

    y_preds = model.predict(Xs)

    APs = []

    for class_idx in range(len(label_maps)):
        recalls = []
        precisions = []
        print(f"class: {label_maps[class_idx]}")
        for confidence_threshold in np.arange(start=0, stop=1, step=0.01):
            y_preds_filtered = []
            for y_pred in y_preds:
                selected_pred = []
                for p in y_pred:
                    obj = p[0]
                    score = p[1]
                    if score >= confidence_threshold and int(obj-1) == class_idx:
                        selected_pred.append(p)

                y_preds_filtered.append(selected_pred)

            TP, TN, FP, FN = 0, 0, 0, 0
            for i, sample in enumerate(samples):
                y_true = [j["bbox"] for j in ys[i]]
                y_pred = np.array(y_preds_filtered[i])

                if len(y_true) == 0 and len(y_pred) != 0:
                    FP += len(y_pred)
                    continue

                if len(y_pred) == 0 and len(y_true) != 0:
                    FN += len(y_true)
                    continue

                y_pred = y_pred[:, 2:6]

                for gt_box in y_true:
                    for y_pred_box in y_pred:
                        iou = bbox_utils.iou(
                            np.expand_dims(gt_box, axis=0),
                            np.expand_dims(y_pred_box, axis=0)
                        )
                        if iou > 0.8:
                            TP += 1
                        else:
                            FP += 1

            if (TP + FN) != 0 and (TP + FP) != 0:
                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
                recalls.append(recall)
                precisions.append(precision)

        precisions.append(1)
        recalls.append(0)

        recalls = np.array(recalls, dtype=np.float)
        precisions = np.array(precisions, dtype=np.float)

        AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
        APs.append(AP)
        print(f"-- ap: {AP}")

    mAP = np.mean(APs)

    print(f"mAP: {mAP}")
