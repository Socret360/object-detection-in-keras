import os
import json
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from networks import QSSD_VGG16
from tensorflow.keras.applications import vgg16
from utils import bbox_utils, qssd_utils, textboxes_utils


def evaluate_qssd_vgg16(config, args):
    print("evaluate_qssd_vgg16")
    input_size = config["model"]["input_size"]
    with open(args.label_maps, "r") as file:
        label_maps = [line.strip("\n") for line in file.readlines()]

    model = QSSD_VGG16(
        config,
        label_maps,
        is_training=False,
        num_predictions=args.num_predictions
    )
    model.load_weights(args.weights)

    images = sorted(list(glob(os.path.join(args.images_dir, "*jpg"))))
    labels = sorted(list(glob(os.path.join(args.images_dir, "*json"))))
    class_id = 5

    Xs = np.zeros((len(images), input_size, input_size, 3), dtype=np.float)
    ys = []

    for i, (image_file, label_file) in enumerate(list(zip(images, labels))):
        print(f"reading sample: {i+1}/{len(images)}")
        image = cv2.imread(image_file)  # read image in bgr format
        input_image = cv2.resize(
            image,
            (input_size, input_size)
        )
        width_scale, height_scale = input_size / \
            image.shape[1], input_size / image.shape[0]
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = vgg16.preprocess_input(input_image)
        Xs[i] = input_image

        with open(label_file, "r") as label_file:
            label = json.load(label_file)
            objects = label["shapes"]
            objs = []
            for obj in objects:
                if obj["label"] == label_maps[class_id-1]:
                    polygon = np.array(obj["points"])
                    tp = polygon.copy()
                    tp[:, 0] = polygon[:, 0]*width_scale
                    tp[:, 1] = polygon[:, 1]*height_scale
                    objs.append({
                        "class": obj["label"],
                        "polygon": tp
                    })
            ys.append(objs)

    y_preds = model.predict(Xs)

    recalls = []
    precisions = []

    for confidence_threshold in np.arange(start=0.2, stop=0.9, step=0.05):
        y_preds_filtered = []
        for y_pred in y_preds:
            selected_pred = []
            for p in y_pred:
                obj = p[0]
                score = p[1]
                if score >= confidence_threshold and obj == class_id:
                    selected_pred.append(p)
            y_preds_filtered.append(selected_pred)

        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(len(images)):
            y_true = textboxes_utils.get_bboxes_from_quads(
                np.array([y["polygon"] for y in ys[i]]))
            y_true = bbox_utils.center_to_corner(y_true)
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

        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        print(f"-- confidence_score: {confidence_threshold}")
        print(f"---- recall: {recall}")
        print(f"---- precision: {precision}")
        recalls.append(recall)
        precisions.append(precision)

    precisions.append(1)
    recalls.append(0)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    plt.plot(recalls, precisions)
    plt.show()
    # exit()
    # print(np.boolean_mask(y_pred[:, :, 0] == (class_id - 1)))
    # print(y_pred[:, :, 1])
