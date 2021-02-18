import cv2
import numpy as np
import tensorflow as tf
from utils.augmentation_utils import photometric, geometric
from utils import ssd_utils, textboxes_utils, one_hot_class_label


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
        self.extra_box_for_ar_1 = model_config["extra_box_for_ar_1"]
        self.default_boxes_config = model_config["default_boxes"]
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
                extra_box_for_ar_1=self.extra_box_for_ar_1
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

    def __augment(self, image):
        augmentations = [
            photometric.random_brightness,
            photometric.random_contrast,
            photometric.random_hue,
            photometric.random_lighting_noise,
            photometric.random_saturation
        ]
        augmented_image = image
        for aug in augmentations:
            augmented_image, _, __ = aug(image=augmented_image)

        return augmented_image

    def __get_data(self, batch):
        X = []
        y = self.input_template.copy()

        for batch_idx, sample_idx in enumerate(batch):
            image_path, label_path = self.samples[sample_idx].split(" ")
            image, quads = textboxes_utils.read_sample(
                image_path=image_path,
                label_path=label_path
            )
            quads = textboxes_utils.sort_quads_vertices(quads)
            bboxes = textboxes_utils.get_bboxes_from_quads(quads)

            if self.perform_augmentation:
                image = self.__augment(image=image)

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

            # image = cv2.resize(np.uint8(image), (self.input_size, self.input_size))

            # print(gt_textboxes.shape)

            # for bbox in bboxes:
            #     cx = bbox[0] * width_scale
            #     cy = bbox[1] * height_scale
            #     width = bbox[2] * width_scale
            #     height = bbox[3] * height_scale
            #     cv2.rectangle(
            #         image,
            #         (int(cx - (width / 2)), int(cy - (height / 2))),
            #         (int(cx + (width / 2)), int(cy + (height / 2))),
            #         (0, 0, 255),
            #         1
            #     )

            # for i, gt_textbox in enumerate(gt_textboxes):
            #     quad = gt_textbox[4:]
            #     q_x1 = quad[0] * self.input_size
            #     q_y1 = quad[1] * self.input_size
            #     q_x2 = quad[2] * self.input_size
            #     q_y2 = quad[3] * self.input_size
            #     q_x3 = quad[4] * self.input_size
            #     q_y3 = quad[5] * self.input_size
            #     q_x4 = quad[6] * self.input_size
            #     q_y4 = quad[7] * self.input_size
            #     bbox = gt_textbox[:4]
            #     print([
            #         bbox[0] * self.input_size,
            #         bbox[1] * self.input_size,
            #         bbox[2] * self.input_size,
            #         bbox[3] * self.input_size,
            #     ])
            #     print(bboxes[i])
            #     cx = bbox[0] * self.input_size
            #     cy = bbox[1] * self.input_size
            #     w = bbox[2] * self.input_size
            #     h = bbox[3] * self.input_size
            #     cv2.polylines(image, np.expand_dims(np.array([
            #         [q_x1, q_y1],
            #         [q_x2, q_y2],
            #         [q_x3, q_y3],
            #         [q_x4, q_y4]
            #     ], dtype=np.int), axis=0), True, (0, 255, 0), 1)
            #     cv2.rectangle(
            #         image,
            #         (int(cx - (w / 2)), int(cy - (h / 2))),
            #         (int(cx + (w / 2)), int(cy + (h / 2))),
            #         (255, 0, 0),
            #         1
            #     )
            #     cv2.circle(
            #         image,
            #         (int(cx), int(cy)),
            #         3,
            #         (0, 0, 255),
            #         3
            #     )
            #     print(gt_textbox.shape)

            # for quad in quads:
            #     cv2.polylines(image, np.expand_dims(np.array(quad, dtype=np.int), axis=0), True, (0, 255, 0), 1)

            # for bbox in bboxes:
            #     cv2.rectangle(
            #         image,
            #         (int(bbox[0] - (bbox[2] / 2)), int(bbox[1] - (bbox[3] / 2))),
            #         (int(bbox[0] + (bbox[2] / 2)), int(bbox[1] + (bbox[3] / 2))),
            #         (255, 0, 0),
            #         1
            #     )

            # cv2.imshow("image", image)
            # if cv2.waitKey(0) == ord('q'):
            #     cv2.destroyAllWindows()

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
