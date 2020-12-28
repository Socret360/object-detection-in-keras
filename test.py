import json
from networks import SSD300_MOBILENET

with open("configs/ssd300_mobilenet.json", "r") as config_file:
    config = json.load(config_file)

with open("/Users/socretlee/Downloads/ssd300_vgg16_voc-07-12_20-classes_118-epochs_label_maps.txt", "r") as file:
    label_maps = [line.strip("\n") for line in file.readlines()]

model = SSD300_MOBILENET(
    config,
    label_maps=label_maps
)
model.summary()
