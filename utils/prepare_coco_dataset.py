import os
import cv2
import argparse
from glob import glob
from xml.dom import minidom
import xml.etree.cElementTree as ET
from pycocotools.coco import COCO

parser = argparse.ArgumentParser(description='Converts the coco dataset to a format suitable for training ssd with this repo.')
parser.add_argument('annotations_file', type=str, help='path to annotations file.')
parser.add_argument('images_dir', type=str, help='path to images dir.')
parser.add_argument('output_dir', type=str, help='path to output dir.')
args = parser.parse_args()

assert os.path.exists(args.annotations_file), "annotations_file does not exist"
assert os.path.exists(args.images_dir), "images_dir does not exist"
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

coco = COCO(args.annotations_file)
categories = coco.cats

print("writing label maps to label_maps.txt")
with open(os.path.join(args.output_dir, "label_maps.txt"), "w+") as label_maps_file:
    for cat_id in categories:
        label_maps_file.write(f"{categories[cat_id]['name']}\n")
print("-- done")

num_samples = 0
print("-- converting coco annotations to xml files")
with open(os.path.join(args.output_dir, "split.txt"), "w+") as split_file:
    images_ids = list(coco.imgs.keys())
    num_images = len(images_ids)
    for i, image_id in enumerate(images_ids):
        print(f"-- image {i+1}/{num_images}")
        annotations = coco.loadAnns(coco.getAnnIds([image_id]))
        image_info = coco.loadImgs([image_id])[0]
        image_filename = image_info["file_name"]
        if len(annotations) == 0:
            print(f"\n---- skipped: {image_filename}\n")
            continue
        xml_root = ET.Element("annotation")
        xml_filename = ET.SubElement(xml_root, "filename").text = image_filename
        xml_size = ET.SubElement(xml_root, "size")
        xml_size_width = ET.SubElement(xml_size, "width").text = str(image_info["width"])
        xml_size_height = ET.SubElement(xml_size, "height").text = str(image_info["height"])
        xml_size_depth = ET.SubElement(xml_size, "depth").text = str(3)
        for annotation in annotations:
            category_id = annotation['category_id']
            bbox = annotation['bbox']
            label = coco.cats[category_id]["name"]
            xml_object = ET.SubElement(xml_root, "object")
            xml_object_name = ET.SubElement(xml_object, "name").text = label
            xml_object_bndbox = ET.SubElement(xml_object, "bndbox")
            xml_object_bndbox_xmin = ET.SubElement(xml_object_bndbox, "xmin").text = str(bbox[0])
            xml_object_bndbox_ymin = ET.SubElement(xml_object_bndbox, "ymin").text = str(bbox[1])
            xml_object_bndbox_xmax = ET.SubElement(xml_object_bndbox, "xmax").text = str(bbox[0] + bbox[2])
            xml_object_bndbox_ymax = ET.SubElement(xml_object_bndbox, "ymax").text = str(bbox[1] + bbox[3])
        xml_tree = ET.ElementTree(xml_root)
        xml_file_name = f"{image_filename[:image_filename.index('.')]}.xml"
        with open(os.path.join(args.output_dir, xml_file_name), "wb+") as xml_file:
            xml_tree.write(xml_file)
            split_file.write(f"{image_filename} {xml_file_name}\n")
            num_samples += 1
    print("-- done")
    print(f"num_samples: {num_samples}")
    print(f"split_file lines: {len(split_file.readlines())}")
    print(f"num files in annotations folder: {len(list(glob(os.path.join(args.output_dir, '*.xml'))))}")
