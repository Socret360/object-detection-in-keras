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

    filename = imname.split("/")[-1]

    shutil.copy(os.path.join(args.images_dir, imname), os.path.join(images_output_dir, filename))

    with open(os.path.join(labels_output_dir, f"{filename[:filename.index('.')]}.txt"), "w") as label_file:
        if len(polygons.shape) == 2:
            word = words[0]
            polygon = np.reshape(polygons, (8,))
            for coord in polygon:
                label_file.write(str(int(coord)))
                label_file.write(",")
            label_file.write(word)
            label_file.write("\n")
        else:
            for i, polygon in enumerate(polygons):
                word = words[i]
                polygon = np.reshape(polygon, (8,))
                for coord in polygon:
                    label_file.write(str(int(coord)))
                    label_file.write(",")
                label_file.write(word)
                label_file.write("\n")

# imnames = ground_truth_file["imnames"][0]
# txts = ground_truth_file["txt"][0]
# wordBBs = ground_truth_file["wordBB"][0]

# for i in range(imnames.shape[0]):
#     imname = imnames[i][0]
#     txt = txts[i]
#     wordBB = wordBBs[i]
#     image = cv2.imread(os.path.join(args.images_dir, imname))

#     print(txt.shape, wordBB.shape)
#     cv2.imshow("image", image)

#     if cv2.waitKey(0) == ord('q'):
#         cv2.destroyAllWindows()
#         exit()

# for j, word in enumerate(txt):
#     filename = word.split(",")[-1]
#     shutil.copy(os.path.join(args.images_dir, word), os.path.join(args.output_dir, filename))
#     with open(os.path.join(args.output_dir, filename), "r")) as label_file:
