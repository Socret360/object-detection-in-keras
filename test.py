import json
from networks import TBPP_VGG16

with open("configs/tbpp300_vgg16.json", "r") as config_file:
    config = json.load(config_file)

model = TBPP_VGG16(config)
model.summary()
