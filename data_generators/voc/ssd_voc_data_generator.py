import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from utils.ssd_utils import generate_default_boxes_for_feature_map, match_gt_boxes_to_default_boxes
from utils import one_hot_class_label


class SSD_VOC_DATA_GENERATOR(tf.keras.utils.Sequence):
    """ Data generator for training SSD networks using with VOC labeled format.
    Args:
    - samples: A list of string representing a data sample (image file path + label file path)
    - config: python dict as read from the config file
    """

    def __init__(self, samples, config):
        self.samples = samples
        #
        self.batch_size = config["training"]["batch_size"]
        self.shuffle = config["training"]["shuffle"]
        self.data_dir = config["training"]["data_dir"]
        #
        self.input_shape = config["model"]["input_shape"]
        self.num_classes = config["model"]["num_classes"] + 1
        self.extra_box_for_ar_1 = config["model"]["extra_box_for_ar_1"]
        self.default_boxes_config = config["model"]["default_boxes"]
        self.label_maps = ["__backgroud__"] + config["model"]["label_maps"]
        self.indices = range(0, len(self.samples))
        #
        assert self.batch_size <= len(self.indices), "batch size must be smaller than the number of samples"
        assert self.input_shape[0] == self.input_shape[1], "input width should be equals to input height"
        self.input_size = min(self.input_shape[0], self.input_shape[1])
        self.input_template = self.__get_input_template()
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_input_template(self):
        scales = np.linspace(
            self.default_boxes_config["min_scale"],
            self.default_boxes_config["max_scale"],
            len(self.default_boxes_config["layers"])
        )
        mbox_conf_layers = []
        mbox_loc_layers = []
        mbox_default_boxes_layers = []
        for i, layer in enumerate(self.default_boxes_config["layers"]):
            layer_default_boxes = generate_default_boxes_for_feature_map(
                feature_map_size=layer["size"],
                image_size=self.input_shape[0],
                offset=layer["offset"],
                scale=scales[i],
                next_scale=scales[i+1] if i+1 <= len(self.default_boxes_config["layers"]) - 1 else 1,
                aspect_ratios=layer["aspect_ratios"],
                variances=self.default_boxes_config["variances"],
                extra_box_for_ar_1=self.extra_box_for_ar_1
            )
            layer_default_boxes = np.reshape(layer_default_boxes, (-1, 8))
            layer_conf = np.zeros((layer_default_boxes.shape[0], self.num_classes))
            layer_conf[:, 0] = 1  # all classes are background by default
            mbox_conf_layers.append(layer_conf)
            mbox_loc_layers.append(np.zeros((layer_default_boxes.shape[0], 4)))
            mbox_default_boxes_layers.append(layer_default_boxes)
        mbox_conf = np.concatenate(mbox_conf_layers, axis=0)
        mbox_loc = np.concatenate(mbox_loc_layers, axis=0)
        mbox_default_boxes = np.concatenate(mbox_default_boxes_layers, axis=0)
        template = np.concatenate([mbox_conf, mbox_loc, mbox_default_boxes], axis=-1)
        template = np.expand_dims(template, axis=0)
        return np.tile(template, (self.batch_size, 1, 1))

    def __get_data(self, batch):
        X = []
        y = self.input_template.copy()

        for sample_idx in batch:
            image_path, label_path = self.samples[sample_idx].split(" ")
            image = cv2.imread(image_path)  # read image in bgr format
            image_height, image_width, _ = image.shape
            height_scale, width_scale = self.input_size/image_height, self.input_size/image_width
            input_img = cv2.resize(image, (self.input_size, self.input_size))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            xml_root = ET.parse(label_path).getroot()
            objects = xml_root.findall("object")
            gt_classes = np.zeros((len(objects), self.num_classes))
            gt_boxes = np.zeros((len(objects), 4))

            for i, obj in enumerate(objects):
                name = obj.find("name").text
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                cx = (((xmin + xmax) / 2) * width_scale) / self.input_size
                cy = (((ymin + ymax) / 2) * height_scale) / self.input_size
                width = (abs(xmax - xmin) * width_scale) / self.input_size
                height = (abs(ymax - ymin) * height_scale) / self.input_size
                gt_boxes[i] = [cx, cy, width, height]
                gt_classes[i] = one_hot_class_label(name, self.label_maps)

            matches = match_gt_boxes_to_default_boxes(
                gt_boxes=gt_boxes,
                default_boxes=y[sample_idx, :, -8:-4],
                threshold=0.5
            )

            matched_gt_boxes = gt_boxes[matches[:, 0]]
            matched_default_boxes = y[sample_idx, :, -8:-4][matches[:, 1]]

            encoded_gt_boxes_cx = ((matched_gt_boxes[:, 0] - matched_default_boxes[:, 0]) / matched_default_boxes[:, 2]) / np.sqrt(self.default_boxes_config["variances"][0])
            encoded_gt_boxes_cy = ((matched_gt_boxes[:, 1] - matched_default_boxes[:, 1]) / matched_default_boxes[:, 3]) / np.sqrt(self.default_boxes_config["variances"][1])
            encoded_gt_boxes_w = np.log(matched_gt_boxes[:, 2] / matched_default_boxes[:, 2]) / np.sqrt(self.default_boxes_config["variances"][2])
            encoded_gt_boxes_h = np.log(matched_gt_boxes[:, 3] / matched_default_boxes[:, 3]) / np.sqrt(self.default_boxes_config["variances"][3])
            encoded_gt_boxes = np.concatenate([
                np.expand_dims(encoded_gt_boxes_cx, axis=-1),
                np.expand_dims(encoded_gt_boxes_cy, axis=-1),
                np.expand_dims(encoded_gt_boxes_w, axis=-1),
                np.expand_dims(encoded_gt_boxes_h, axis=-1),
            ], axis=-1)

            y[sample_idx, matches[:, 1], self.num_classes + 1: self.num_classes + 5] = encoded_gt_boxes  # set ground truth boxes to matched default boxes
            y[sample_idx, matches[:, 1], 0: self.num_classes] = gt_classes[matches[:, 0]]  # set class scores label
            X.append(input_img)

        return X, y
