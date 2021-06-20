import cv2
import numpy as np
import tensorflow as tf
from utils import ssd_utils, textboxes_utils, one_hot_class_label, augmentation_utils
from time import time


class TBPP_DATA_GENERATOR(tf.keras.utils.Sequence):
    """"""

    def __init__(
        self,
        samples,
        config,
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
        self.default_boxes_config = model_config["default_boxes"]
        self.extra_box_for_ar_1 = self.default_boxes_config["extra_box_for_ar_1"]
        self.clip_default_boxes = self.default_boxes_config["clip_boxes"]
        self.label_maps = ["__backgroud__", "text"]
        self.num_classes = len(self.label_maps)
        self.indices = range(0, len(self.samples))
        #
        assert self.batch_size <= len(self.indices), "batch size must be smaller than the number of samples"
        self.input_size = model_config["input_size"]
        self.input_template = self.__get_input_template()
        self.perform_augmentation = augment
        self.process_input_fn = process_input_fn
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
            layer_default_boxes = ssd_utils.generate_default_boxes_for_feature_map(
                feature_map_size=layer["size"],
                image_size=self.input_size,
                offset=layer["offset"],
                scale=scales[i],
                next_scale=scales[i+1] if i+1 <= len(self.default_boxes_config["layers"]) - 1 else 1,
                aspect_ratios=layer["aspect_ratios"],
                variances=self.default_boxes_config["variances"],
                extra_box_for_ar_1=self.extra_box_for_ar_1,
                clip_boxes=self.clip_default_boxes
            )
            layer_default_boxes = np.reshape(layer_default_boxes, (-1, 8))
            layer_conf = np.zeros((layer_default_boxes.shape[0], self.num_classes))
            layer_conf[:, 0] = 1  # all classes are background by default
            mbox_conf_layers.append(layer_conf)
            mbox_loc_layers.append(np.zeros((layer_default_boxes.shape[0], 12)))
            mbox_default_boxes_layers.append(layer_default_boxes)
        mbox_conf = np.concatenate(mbox_conf_layers, axis=0)
        mbox_loc = np.concatenate(mbox_loc_layers, axis=0)
        mbox_default_boxes = np.concatenate(mbox_default_boxes_layers, axis=0)
        template = np.concatenate([mbox_conf, mbox_loc, mbox_default_boxes], axis=-1)
        template = np.expand_dims(template, axis=0)
        return np.tile(template, (self.batch_size, 1, 1))

    def __augment(self, image, quads, classes):
        augmented_image, augmented_quads, augmented_classes = augmentation_utils.random_brightness(
            image=image,
            bboxes=quads,
            classes=classes,
        )
        augmented_image, augmented_quads, augmented_classes = augmentation_utils.random_contrast(
            image=augmented_image,
            bboxes=augmented_quads,
            classes=augmented_classes,
        )
        augmented_image, augmented_quads, augmented_classes = augmentation_utils.random_hue(
            image=augmented_image,
            bboxes=augmented_quads,
            classes=augmented_classes,
        )
        augmented_image, augmented_quads, augmented_classes = augmentation_utils.random_lighting_noise(
            image=augmented_image,
            bboxes=augmented_quads,
            classes=augmented_classes,
        )
        augmented_image, augmented_quads, augmented_classes = augmentation_utils.random_saturation(
            image=augmented_image,
            bboxes=augmented_quads,
            classes=augmented_classes,
        )
        augmented_image, augmented_quads, augmented_classes = augmentation_utils.random_horizontal_flip_quad(
            image=augmented_image,
            quads=augmented_quads,
            classes=augmented_classes,
        )
        augmented_image, augmented_quads, augmented_classes = augmentation_utils.random_vertical_flip_quad(
            image=augmented_image,
            quads=augmented_quads,
            classes=augmented_classes,
        )
        return augmented_image, augmented_quads, augmented_classes

    def __get_data(self, batch):
        X = []
        y = self.input_template.copy()

        for batch_idx, sample_idx in enumerate(batch):
            image_path, label_path = self.samples[sample_idx].split(" ")
            image, quads = textboxes_utils.read_sample(
                image_path=image_path,
                label_path=label_path
            )

            if self.perform_augmentation:
                image, quads, _ = self.__augment(
                    image=image,
                    quads=quads,
                    classes=None,
                )

            quads = textboxes_utils.sort_quads_vertices(quads)
            bboxes = textboxes_utils.get_bboxes_from_quads(quads)
            image_height, image_width, _ = image.shape
            height_scale, width_scale = self.input_size/image_height, self.input_size/image_width
            input_img = cv2.resize(np.uint8(image), (self.input_size, self.input_size))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = self.process_input_fn(input_img)

            gt_classes = np.zeros((quads.shape[0], self.num_classes))
            gt_textboxes = np.zeros((quads.shape[0], 12))
            default_boxes = y[batch_idx, :, -8:]

            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                quad = quads[i]
                cx = bbox[0] * width_scale / self.input_size
                cy = bbox[1] * height_scale / self.input_size
                width = bbox[2] * width_scale / self.input_size
                height = bbox[3] * height_scale / self.input_size
                q_x1 = quad[0, 0] * width_scale / self.input_size
                q_y1 = quad[0, 1] * height_scale / self.input_size
                q_x2 = quad[1, 0] * width_scale / self.input_size
                q_y2 = quad[1, 1] * height_scale / self.input_size
                q_x3 = quad[2, 0] * width_scale / self.input_size
                q_y3 = quad[2, 1] * height_scale / self.input_size
                q_x4 = quad[3, 0] * width_scale / self.input_size
                q_y4 = quad[3, 1] * height_scale / self.input_size
                gt_textboxes[i, :4] = [cx, cy, width, height]
                gt_textboxes[i, 4:] = [q_x1, q_y1, q_x2, q_y2, q_x3, q_y3, q_x4, q_y4]
                gt_classes[i] = [0, 1]

            matches, neutral_boxes = ssd_utils.match_gt_boxes_to_default_boxes(
                gt_boxes=gt_textboxes[:, :4],
                default_boxes=default_boxes[:, :4],
                match_threshold=self.match_threshold,
                neutral_threshold=self.neutral_threshold
            )

            # set matched ground truth boxes to default boxes with appropriate class
            y[batch_idx, matches[:, 1], self.num_classes: self.num_classes + 12] = gt_textboxes[matches[:, 0]]
            y[batch_idx, matches[:, 1], 0: self.num_classes] = gt_classes[matches[:, 0]]  # set class scores label
            # set neutral ground truth boxes to default boxes with appropriate class
            y[batch_idx, neutral_boxes[:, 1], self.num_classes: self.num_classes + 12] = gt_textboxes[neutral_boxes[:, 0]]
            y[batch_idx, neutral_boxes[:, 1], 0: self.num_classes] = np.zeros((self.num_classes))  # neutral boxes have a class vector of all zeros
            # encode the bounding boxes
            y[batch_idx] = textboxes_utils.encode_textboxes(y[batch_idx])
            X.append(input_img)

        X = np.array(X, dtype=np.float)

        return X, y
