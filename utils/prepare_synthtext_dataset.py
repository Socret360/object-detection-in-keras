import os
import cv2
import numpy as np
import argparse
import shutil
from scipy import io
from glob import glob

parser = argparse.ArgumentParser(description='Converts the synthtext dataset to a format suitable for training textboxes plus plus with this repo.')
parser.add_argument('annotations_file', type=str, help='path to annotations file.')
parser.add_argument('images_dir', type=str, help='path to images dir.')
parser.add_argument('output_dir', type=str, help='path to output dir.')
args = parser.parse_args()

assert os.path.exists(args.annotations_file), "annotations_file does not exist"
assert os.path.exists(args.images_dir), "images_dir does not exist"

images_output_dir = os.path.join(args.output_dir, "images")
labels_output_dir = os.path.join(args.output_dir, "labels")

os.makedirs(images_output_dir, exist_ok=True)
os.makedirs(labels_output_dir, exist_ok=True)

ground_truth_file = io.loadmat(args.annotations_file)


def clip_polygon(p, image):
    image_height, image_width, _ = image.shape
    polygon = p.copy()
    for n in [0, 2, 4, 6]:
        if polygon[n] < 0:
            polygon[n] = 0
        elif polygon[n] > image_width:
            polygon[n] = image_width
    for n in [1, 3, 5, 7]:
        if polygon[n] < 0:
            polygon[n] = 0
        elif polygon[n] > image_height:
            polygon[n] = image_height
    return polygon


with open(os.path.join(args.output_dir, "samples.txt"), "w") as samples_file:
    for img_id in range(ground_truth_file["imnames"].shape[-1]):
        print(f"image: {img_id+1}/{ground_truth_file['imnames'].shape[-1]}")
        imname = ground_truth_file["imnames"][0][img_id][0]
        texts = ground_truth_file["txt"][0][img_id]
        wordBboxes = ground_truth_file["wordBB"][0]
        polygons = np.concatenate(
            [
                np.expand_dims(wordBboxes[img_id][0].transpose(), axis=-1),
                np.expand_dims(wordBboxes[img_id][1].transpose(), axis=-1),
            ],
            axis=-1
        )

        words = []
        for word in texts:
            for i in word.split("\n"):
                for j in i.split(" "):
                    if j != "":
                        words.append(j)

        filename = os.path.basename(imname)
        sample = f"{filename} {filename[:filename.index('.')]}.txt"

        shutil.copy(os.path.join(args.images_dir, imname), os.path.join(images_output_dir, filename))
        with open(os.path.join(labels_output_dir, f"{filename[:filename.index('.')]}.txt"), "w") as label_file:
            image = cv2.imread(os.path.join(images_output_dir, filename))
            if len(polygons.shape) == 2:
                word = words[0]
                polygon = np.reshape(polygons, (8,))
                polygon = clip_polygon(polygon, image)

                for coord in polygon:
                    label_file.write(str(float(coord)))
                    label_file.write(",")
                label_file.write(word)
                label_file.write("\n")
            else:
                for i, polygon in enumerate(polygons):
                    word = words[i]
                    polygon = np.reshape(polygon, (8,))
                    polygon = clip_polygon(polygon, image)

                    for coord in polygon:
                        label_file.write(str(float(coord)))
                        label_file.write(",")
                    label_file.write(word)
                    label_file.write("\n")

        samples_file.write(sample)
        samples_file.write("\n")
