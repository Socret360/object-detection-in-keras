from utils.augmentation_utils.resize_to_fixed_size import resize_to_fixed_size
import os
import cv2
import numpy as np
import tensorflow as tf
from utils import ssd_utils, one_hot_class_label, augmentation_utils


class SSD_DATA_GENERATOR(tf.keras.utils.Sequence):
    """ Data generator for training SSD networks using with VOC labeled format.

    Args:
        - samples: A list of string representing a data sample (image file path + label file path)
        - config: python dict as read from the config file
        - label_maps: A list of classes in the dataset.
        - shuffle: Whether or not to shuffle the batch.
        - batch_size: The size of each batch
        - augment: Whether or not to augment the training samples.
        - process_input_fn: A function to preprocess input image before feeding into the network
    """

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
        self.default_boxes_config = model_config["default_boxes"]
        self.extra_box_for_ar_1 = self.default_boxes_config["extra_box_for_ar_1"]
        self.clip_default_boxes = self.default_boxes_config["clip_boxes"]
        self.label_maps = ["__backgroud__"] + label_maps
        self.num_classes = len(self.label_maps)
        self.indices = range(0, len(self.samples))
        #
        assert self.batch_size <= len(self.indices), "batch size must be smaller than the number of samples"
        self.input_size = model_config["input_size"]
        self.input_template = self.__get_input_template()
        self.perform_augmentation = augment
        self.process_input_fn = process_input_fn

        self.augmentations = [
            augmentation_utils.random_expand(p=1, min_ratio=1, max_ratio=4),
            augmentation_utils.random_crop(p=1),
            augmentation_utils.resize_to_fixed_size(width=self.input_size, height=self.input_size),
            augmentation_utils.random_horizontal_flip(p=0.5),
            augmentation_utils.random_brightness(p=0.5),
            augmentation_utils.random_contrast(p=0.5),
            augmentation_utils.random_hue(p=0.5),
            augmentation_utils.random_saturation(p=0.5),
        ] if self.perform_augmentation else [
            augmentation_utils.resize_to_fixed_size(width=self.input_size, height=self.input_size),
        ]
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
        if self.shuffle:
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
                clip_boxes=self.clip_default_boxes,
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

    def __augment(self, image, bboxes, classes):
        augmented_image, augmented_bboxes, augmented_classes = image, bboxes, classes
        for aug in self.augmentations:
            augmented_image, augmented_bboxes, augmented_classes = aug(
                image=augmented_image,
                bboxes=augmented_bboxes,
                classes=augmented_classes
            )

        return augmented_image, augmented_bboxes, augmented_classes

    def __get_data(self, batch):
        X = []
        y = self.input_template.copy()

        for batch_idx, sample_idx in enumerate(batch):
            image_path, label_path = self.samples[sample_idx].split(" ")
            # print(f"image: {os.path.basename(image_path)}")
            image, bboxes, classes = ssd_utils.read_sample(
                image_path=image_path,
                label_path=label_path
            )

            image, bboxes, classes = self.__augment(
                image=image,
                bboxes=bboxes,
                classes=classes
            )

            input_img = np.uint8(image)

            # for i, bbox in enumerate(bboxes):
            #     cv2.rectangle(
            #         input_img,
            #         (int(bbox[0]), int(bbox[1])),
            #         (int(bbox[2]), int(bbox[3])),
            #         (0, 255, 0),
            #         2
            #     )
            #     print(f"-- {classes[i]}: {bbox}, area: {int(abs(bbox[0] - bbox[2]) * abs(bbox[1] - bbox[3]))}")

            # cv2.imshow("sample", input_img)

            # if cv2.waitKey(0) == ord('q'):
            #     cv2.destroyAllWindows()
            #     continue

            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = self.process_input_fn(input_img)

            gt_classes = np.zeros((bboxes.shape[0], self.num_classes))
            gt_boxes = np.zeros((bboxes.shape[0], 4))
            default_boxes = y[batch_idx, :, -8:]

            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                cx = ((bbox[0] + bbox[2]) / 2) / self.input_size
                cy = ((bbox[1] + bbox[3]) / 2) / self.input_size
                width = abs(bbox[2] - bbox[0]) / self.input_size
                height = abs(bbox[3] - bbox[1]) / self.input_size
                gt_boxes[i] = [cx, cy, width, height]
                gt_classes[i] = one_hot_class_label(classes[i], self.label_maps)

            matches, neutral_boxes = ssd_utils.match_gt_boxes_to_default_boxes(
                gt_boxes=gt_boxes,
                default_boxes=default_boxes[:, :4],
                match_threshold=self.match_threshold,
                neutral_threshold=self.neutral_threshold
            )
            # set matched ground truth boxes to default boxes with appropriate class
            y[batch_idx, matches[:, 1], self.num_classes: self.num_classes + 4] = gt_boxes[matches[:, 0]]
            y[batch_idx, matches[:, 1], 0: self.num_classes] = gt_classes[matches[:, 0]]  # set class scores label
            # set neutral ground truth boxes to default boxes with appropriate class
            y[batch_idx, neutral_boxes[:, 1], self.num_classes: self.num_classes + 4] = gt_boxes[neutral_boxes[:, 0]]
            y[batch_idx, neutral_boxes[:, 1], 0: self.num_classes] = np.zeros((self.num_classes))  # neutral boxes have a class vector of all zeros
            # encode the bounding boxes
            y[batch_idx] = ssd_utils.encode_bboxes(y[batch_idx])
            X.append(input_img)

        X = np.array(X, dtype=np.float)

        # exit()

        return X, y
