import os
import numpy as np


def get_num_quads(label_file):
    """"""
    label_path = label_file.strip("\n")
    assert os.path.exists(label_path), "Label file does not exist."

    with open(label_path, "r") as label_file:
        temp_labels = label_file.readlines()

        num_labels = 0

        for label in temp_labels:
            label = label.strip("\ufeff").strip("\n")
            label = label.split(",")

            if len(label[:-1]) != 8:
                continue

            num_labels += 1

    return num_labels
