import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from models import SSD300_VGG16_ORIGINAL
from data_generators import SSD_VOC_DATA_GENERATOR
from utils import generate_default_boxes_for_feature_map, match_gt_boxes_to_default_boxes

with open("configs/ssd300_vgg16_original.json") as config_file:
    config = json.load(config_file)

# image = cv2.imread("data/test.jpg")
# image_height, image_width, _ = image.shape
# image = cv2.resize(image, (300, 300))
# xml_root = ET.parse("data/test.xml").getroot()
# height_scale, width_scale = 300/image_height, 300/image_width
# objects = xml_root.findall("object")

# scales = np.linspace(0.1, 0.9, 6)
# default_boxes = []
# layers = config["model"]["default_boxes"]["layers"]

# scales = np.linspace(
#     config["model"]["default_boxes"]["min_scale"],
#     config["model"]["default_boxes"]["max_scale"],
#     len(config["model"]["default_boxes"]["layers"])
# )
# mbox_conf_layers = []
# mbox_loc_layers = []
# mbox_default_boxes_layers = []
# for i, layer in enumerate(layers):
#     layer_default_boxes = generate_default_boxes_for_feature_map(
#         feature_map_size=layer["size"],
#         image_size=config["model"]["input_shape"][0],
#         offset=layer["offset"],
#         scale=scales[i],
#         next_scale=scales[i+1] if i+1 <= len(layers) - 1 else 1,
#         aspect_ratios=layer["aspect_ratios"],
#         variances=config["model"]["default_boxes"]["variances"],
#         normalize_coords=config["model"]["normalize_coords"],
#         extra_box_for_ar_1=config["model"]["extra_box_for_ar_1"]
#     )
#     layer_default_boxes_reshape = np.reshape(layer_default_boxes, (-1, 8))
#     mbox_conf_layers.append(np.zeros((layer_default_boxes_reshape.shape[0], config["model"]["num_classes"])))
#     mbox_loc_layers.append(np.zeros((layer_default_boxes_reshape.shape[0], 4)))
#     mbox_default_boxes_layers.append(layer_default_boxes_reshape)
# mbox_conf = np.concatenate(mbox_conf_layers, axis=0)
# mbox_loc = np.concatenate(mbox_loc_layers, axis=0)
# mbox_default_boxes = np.concatenate(mbox_default_boxes_layers, axis=0)
# res = np.concatenate([mbox_conf, mbox_loc, mbox_default_boxes], axis=-1)
# res = np.tile(np.expand_dims(res, axis=0), (config["training"]["batch_size"], 1, 1))
# print(res.shape)


# for i, layer in enumerate(layers):
#     default_boxes_layer = generate_default_boxes_for_feature_map(
#         feature_map_size=layer["size"],
#         image_size=config["model"]["input_shape"][0],
#         offset=layer["offset"],
#         scale=scales[i],
#         next_scale=scales[i+1] if i+1 <= len(layers) - 1 else 1,
#         aspect_ratios=layer["aspect_ratios"],
#         variances=config["model"]["default_boxes"]["variances"],
#         normalize_coords=config["model"]["normalize_coords"],
#         extra_box_for_ar_1=config["model"]["extra_box_for_ar_1"]
#     )
#     default_boxes_layer = np.reshape(default_boxes_layer, (-1, 8))
#     default_boxes.append(default_boxes_layer)
# default_boxes = np.concatenate(default_boxes, axis=0)
# default_boxes[:, [0, 2]] *= config["model"]["input_shape"][0]
# default_boxes[:, [1, 3]] *= config["model"]["input_shape"][0]

# gt_truth_boxes = np.zeros((len(objects), 4))

# for i, obj in enumerate(objects):
#     name = obj.find("name").text
#     bndbox = obj.find("bndbox")
#     xmin = (int(bndbox.find("xmin").text) * width_scale)
#     ymin = (int(bndbox.find("ymin").text) * height_scale)
#     xmax = (int(bndbox.find("xmax").text) * width_scale)
#     ymax = (int(bndbox.find("ymax").text) * height_scale)
#     gt_truth_boxes[i] = [xmin, ymin, xmax, ymax]

# matches = match_gt_boxes_to_default_boxes(
#     gt_truth_boxes,
#     default_boxes[:, :4]
# ).astype(int)

# model = SSD300_VGG16_ORIGINAL(config=config)
# model.summary()

# print(default_boxes.shape)

# cv2.imshow("image", image)

# if cv2.waitKey(0) == ord('q'):
#     cv2.destroyAllWindows()

data_generator = SSD_VOC_DATA_GENERATOR(
    samples=["data/test.jpg data/test.xml"],
    config=config
)

limit = 1

for i, (batch_x, batch_y) in enumerate(data_generator):
    print(f"batch {i+1}")
    for j in range(len(batch_x)):
        print(f"-- item {j}")
    if i >= limit:
        break
