import cv2
import numpy as np
import tensorflow as tf
from utils import ssd_utils, qssd_utils, one_hot_class_label, augmentation_utils, textboxes_utils
from time import time


class QSSD_DATA_GENERATOR(tf.keras.utils.Sequence):
    def __init__(
        self,
        samples,
        config,
        label_maps,
        shuffle,
        batch_size,
        augment,
        process_input_fn,
    ):
        training_config = config["training"]
        model_config = config["model"]
        self.samples = samples
        self.model_name = model_config["name"]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.match_threshold = training_config["match_threshold"]
        self.neutral_threshold = training_config["neutral_threshold"]
        self.extra_box_for_ar_1 = model_config["extra_box_for_ar_1"]
        self.default_quads = model_config["default_quads"]
        self.label_maps = ["__backgroud__"] + label_maps
        self.num_classes = len(self.label_maps)
        self.indices = range(0, len(self.samples))
        #
        assert self.batch_size <= len(
            self.indices), "batch size must be smaller than the number of samples"
        self.input_size = model_config["input_size"]
        self.input_template = self.__get_input_template()
        self.perform_augmentation = augment
        self.process_input_fn = process_input_fn
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index *
                           self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_input_template(self):
        scales = np.linspace(
            self.default_quads["min_scale"],
            self.default_quads["max_scale"],
            len(self.default_quads["layers"])
        )
        mbox_conf_layers = []
        mbox_loc_layers = []
        mbox_default_boxes_layers = []
        for i, layer in enumerate(self.default_quads["layers"]):
            layer_default_quads = qssd_utils.generate_default_quads_for_feature_map(
                feature_map_size=layer["size"],
                image_size=self.input_size,
                offset=layer["offset"],
                scale=scales[i],
                next_scale=scales[i+1] if i +
                1 <= len(self.default_quads["layers"]) - 1 else 1,
                aspect_ratios=layer["aspect_ratios"],
                angles=layer["angles"],
                variances=self.default_quads["variances"],
                extra_box_for_ar_1=self.extra_box_for_ar_1
            )
            layer_default_quads = np.reshape(layer_default_quads, (-1, 16))
            layer_conf = np.zeros(
                (layer_default_quads.shape[0], self.num_classes))
            layer_conf[:, 0] = 1  # all classes are background by default
            mbox_conf_layers.append(layer_conf)
            mbox_loc_layers.append(
                np.zeros((layer_default_quads.shape[0], 8)))
            mbox_default_boxes_layers.append(layer_default_quads)
        mbox_conf = np.concatenate(mbox_conf_layers, axis=0)
        mbox_loc = np.concatenate(mbox_loc_layers, axis=0)
        mbox_default_boxes = np.concatenate(mbox_default_boxes_layers, axis=0)
        template = np.concatenate(
            [mbox_conf, mbox_loc, mbox_default_boxes], axis=-1)
        template = np.expand_dims(template, axis=0)
        return np.tile(template, (self.batch_size, 1, 1))

    def __get_data(self, batch):
        X = []
        y = self.input_template.copy()

        for batch_idx, sample_idx in enumerate(batch):
            image_path, label_path = self.samples[sample_idx].split(" ")
            image, quads, classes = qssd_utils.read_sample(
                image_path=image_path,
                label_path=label_path
            )
            quads = textboxes_utils.sort_quads_vertices(np.reshape(quads, (-1, 4, 2)))
            quads = np.reshape(quads, (-1, 8))
            image_height, image_width, _ = image.shape
            height_scale, width_scale = self.input_size / \
                image_height, self.input_size/image_width
            input_img = cv2.resize(
                np.uint8(image), (self.input_size, self.input_size))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = self.process_input_fn(input_img)

            gt_classes = np.zeros((quads.shape[0], self.num_classes))
            gt_quads = np.zeros((quads.shape[0], 8))
            default_boxes = y[batch_idx, :, -16:]

            gt_quads[:, [0, 2, 4, 6]] = quads[:, [0, 2, 4, 6]] * width_scale / self.input_size
            gt_quads[:, [1, 3, 5, 7]] = quads[:, [1, 3, 5, 7]] * height_scale / self.input_size

            for i in range(quads.shape[0]):
                gt_classes[i] = one_hot_class_label(
                    classes[i], self.label_maps)

            matches, neutral_boxes = qssd_utils.match_gt_quads_to_default_quads(
                gt_quads=gt_quads[:, :8],
                default_boxes=default_boxes[:, :8],
                match_threshold=self.match_threshold,
                neutral_threshold=self.neutral_threshold
            )

            # set matched ground truth boxes to default boxes with appropriate class
            y[batch_idx, matches[:, 1], self.num_classes: self.num_classes +
                8] = gt_quads[matches[:, 0]]
            # set class scores label
            y[batch_idx, matches[:, 1], 0: self.num_classes] = gt_classes[matches[:, 0]]
            # set neutral ground truth boxes to default boxes with appropriate class
            y[batch_idx, neutral_boxes[:, 1], self.num_classes: self.num_classes +
                8] = gt_quads[neutral_boxes[:, 0]]
            y[batch_idx, neutral_boxes[:, 1], 0: self.num_classes] = np.zeros(
                (self.num_classes))  # neutral boxes have a class vector of all zeros
            # encode the bounding boxes
            y[batch_idx] = qssd_utils.encode_qboxes(y[batch_idx])
            X.append(input_img)

        X = np.array(X, dtype=np.float)

        return X, y
