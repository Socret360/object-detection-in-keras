import cv2
import os
from pycocotools.coco import COCO

annotations_path = "/Users/socretlee/Downloads/fast-ai-coco/annotations/"
images_path = "/Users/socretlee/Downloads/fast-ai-coco/train2017/"
coco = COCO(os.path.join(annotations_path, "instances_train2017.json"))

cats = coco.cats

for image_id in coco.imgs.keys():
    image_info = coco.loadImgs([image_id])
    annotations = coco.loadAnns(coco.getAnnIds([image_id]))
    image = cv2.imread(os.path.join(images_path, image_info[0]["file_name"]))
    for annotation in annotations:
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        cv2.rectangle(
            image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
            (0, 255, 0),
            1
        )
        cv2.putText(
            image,
            coco.cats[category_id]["name"],
            (int(bbox[0]), int(bbox[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1
        )

    cv2.imshow('image', image)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

    exit()
# import json
# from networks import SSD300_MOBILENET

# with open("configs/ssd300_mobilenet.json", "r") as config_file:
#     config = json.load(config_file)

# with open("/Users/socretlee/Downloads/ssd300_vgg16_voc-07-12_20-classes_118-epochs_label_maps.txt", "r") as file:
#     label_maps = [line.strip("\n") for line in file.readlines()]

# model = SSD300_MOBILENET(
#     config,
#     label_maps=label_maps
# )
# model.summary()
