from utils import textboxes_utils
image, quads = textboxes_utils.read_sample(
    "output/cocotextv2/val/images/COCO_train2014_000000581563.jpg",
    "output/cocotextv2/val/labels/COCO_train2014_000000581563.txt"
)

print(quads)
