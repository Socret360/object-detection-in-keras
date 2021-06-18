import enum
from glob import glob
import os

from matplotlib import pyplot as plt
from utils import bbox_utils, ssd_utils
import csv
import cv2
import pandas as pd
import numpy as np
from networks.ssd_vgg16 import SSD_VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


def ssd_vgg16(config, args):
    """"""
    # input_size = config["model"]["input_size"]

    with open(args.label_maps, "r") as file:
        label_maps = [line.strip("\n") for line in file.readlines()]

    # model = SSD_VGG16(
    #     config,
    #     label_maps,
    #     is_training=False,
    #     num_predictions=args.num_predictions
    # )

    # model.load_weights(args.weights)

    # with open(args.split_file, "r") as split_file:
    #     samples = [i.strip("\n") for i in split_file.readlines()]

    # Xs, ys = np.zeros((len(samples), input_size, input_size, 3), dtype=np.float), []

    # print("-- creating csv files to store predictions")
    # header = ['image', 'bbox_true', 'bbox_pred', 'class_true', 'class_pred', 'confidence_score', 'iou']

    # with open(os.path.join(args.output_dir, "predictions.csv"), "w") as predictions_file:
    #     predictions_file_writer = csv.writer(predictions_file)
    #     predictions_file_writer.writerow(header)
    #     results = []
    #     for i, sample in enumerate(samples):
    #         print(f"reading sample: {i+1}/{len(samples)}")
    #         image_file, label_file = sample.split(" ")
    #         image, bboxes, classes = ssd_utils.read_sample(
    #             image_path=os.path.join(args.images_dir, image_file),
    #             label_path=os.path.join(args.labels_dir, label_file),
    #         )
    #         image_height, image_width, _ = image.shape
    #         height_scale, width_scale = input_size/image_height, input_size/image_width
    #         input_img = cv2.resize(np.uint8(image), (input_size, input_size))
    #         input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    #         input_img = preprocess_input(input_img)
    #         detections = model.predict(np.array([input_img]))[0]

    #         for i, detection in enumerate(detections):
    #             for j, gt in enumerate(bboxes):
    #                 gt[[0, 2]] *= width_scale
    #                 gt[[1, 3]] *= height_scale
    #                 predictions_file_writer.writerow([
    #                     image_file.split("/")[-1],
    #                     j,
    #                     i,
    #                     classes[j],
    #                     label_maps[int(detection[0]) - 1],
    #                     detection[1],
    #                     bbox_utils.iou(
    #                         np.expand_dims(detection[2:], axis=0),
    #                         np.expand_dims(gt, axis=0)
    #                     )[0],
    #                 ])

    print("-- read predictions file")
    predictions = pd.read_csv(os.path.join(args.output_dir, "cp_105_predictions.csv"))
    predictions["tp"] = 0
    predictions["fp"] = 0
    predictions["bbox_true_max_iou"] = 0

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    aps = []

    for classname in label_maps:
        print(f"-- class: {classname}")
        same_class = (predictions['class_pred'] == classname) & (predictions['class_true'] == classname)

        predictions_for_class = predictions[same_class].copy()
        all_ground_truth = len(predictions_for_class.groupby(["image", "bbox_true"]).size())
        for image_file in predictions_for_class["image"].unique():
            same_image_file = predictions_for_class["image"] == image_file
            for bbox_true_idx in predictions_for_class.loc[same_class & same_image_file, "bbox_true"].unique():
                same_bbox_true = predictions_for_class["bbox_true"] == bbox_true_idx
                predictions_for_class.loc[same_image_file & same_bbox_true, "bbox_true_max_iou"] = predictions_for_class.loc[same_image_file & same_bbox_true, "iou"].max()

        pass_iou_threshold = predictions_for_class["iou"] > args.iou_threshold
        is_max_iou = predictions_for_class["bbox_true_max_iou"] == predictions_for_class["iou"]
        failed_iou_threshold = predictions_for_class["iou"] < args.iou_threshold
        not_max_iou = predictions_for_class["bbox_true_max_iou"] != predictions_for_class["iou"]

        predictions_for_class.loc[pass_iou_threshold & is_max_iou, "tp"] = 1
        predictions_for_class.loc[failed_iou_threshold | not_max_iou, "fp"] = 1
        predictions_for_class = predictions_for_class.sort_values("confidence_score", ascending=False).copy()

        precisions = []
        recalls = []
        for confidence_score_threshold in np.arange(0, 1.1, 0.1):
            predictions_for_threshold = predictions_for_class[predictions_for_class["confidence_score"] > (1-confidence_score_threshold)].copy()
            if (predictions_for_threshold.shape[0] > 0):
                tp = sum(predictions_for_threshold["tp"])
                precision = tp / len(predictions_for_threshold)
                recall = tp / all_ground_truth
                precisions.append(precision)
                recalls.append(recall)

        precisions = np.array(precisions)
        recalls = np.array(recalls)

        interpolated_precisions = []
        interpolated_recalls = []

        for recall in np.arange(0, 1.1, 0.1):
            interpolated_recalls.append(recall)
            temp = recalls[np.where(recalls > recall)]
            if len(temp) == 0:
                interpolated_precisions.append(0)
            else:
                interpolated_precisions.append(max(temp))

        interpolated_precisions = np.array(interpolated_precisions)
        interpolated_recalls = np.array(interpolated_recalls)

        ap = interpolated_precisions.mean()
        print(f"---- ap: {ap}")
        aps.append(ap)

        ax1.plot(recalls, precisions, label=classname)
        ax2.plot(interpolated_recalls, interpolated_precisions, label=classname, linestyle='dashed')

    aps = np.array(aps)
    print(f"-- mAP: {aps.mean()}")

    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax1.title.set_text("PR Curve")
    ax2.title.set_text("Average Precision")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    plt.show()
