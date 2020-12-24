import cv2
import json
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from networks import SSD300_VGG16

with open("configs/ssd300_vgg16.json") as config_file:
    config = json.load(config_file)

num_classes = config["model"]["num_classes"] + 1
label_maps = ["__backgroud__"] + config["model"]["label_maps"]
input_size, __, __ = config["model"]["input_shape"]
model = SSD300_VGG16(config, is_training=False)
model.load_weights(config["inference"]["weights_path"], by_name=True)
image = cv2.imread("data/test.jpg")
image_height, image_width, _ = image.shape
height_scale, width_scale = input_size/image_height, input_size/image_width
image = cv2.resize(image, (input_size, input_size))
t_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = preprocess_input(image)
image = np.expand_dims(image, axis=0)
y_pred = model.predict(image)

#
tf_img_width = 300
tf_img_height = 300
tf_confidence_thresh = 0.01
tf_nms_max_output_size = 400
iou_threshold = 0.45
tf_top_k = 10


def decode(y_pred):
    # Convert anchor box offsets to image offsets.
    cx = y_pred[..., -12] * y_pred[..., -4] * y_pred[..., -6] + y_pred[..., -8]  # cx = cx_pred * cx_variance * w_anchor + cx_anchor
    cy = y_pred[..., -11] * y_pred[..., -3] * y_pred[..., -5] + y_pred[..., -7]  # cy = cy_pred * cy_variance * h_anchor + cy_anchor
    w = tf.exp(y_pred[..., -10] * tf.sqrt(y_pred[..., -2])) * y_pred[..., -6]  # w = exp(w_pred * variance_w) * w_anchor
    h = tf.exp(y_pred[..., -9] * tf.sqrt(y_pred[..., -1])) * y_pred[..., -5]  # h = exp(h_pred * variance_h) * h_anchor

    # Convert 'centroids' to 'corners'.
    xmin = cx - 0.5 * w
    ymin = cy - 0.5 * h
    xmax = cx + 0.5 * w
    ymax = cy + 0.5 * h

    xmin = tf.expand_dims(xmin * tf_img_width, axis=-1)
    ymin = tf.expand_dims(ymin * tf_img_height, axis=-1)
    xmax = tf.expand_dims(xmax * tf_img_width, axis=-1)
    ymax = tf.expand_dims(ymax * tf_img_height, axis=-1)

    # Concatenate the one-hot class confidences and the converted box coordinates to form the decoded predictions tensor.
    y_pred = tf.concat(values=[y_pred[..., :-12], xmin, ymin, xmax, ymax], axis=-1)

    #####################################################################################
    # 2. Perform confidence thresholding, per-class non-maximum suppression, and
    #    top-k filtering.
    #####################################################################################

    batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32
    n_boxes = tf.shape(y_pred)[1]
    n_classes = y_pred.shape[2] - 4
    class_indices = tf.range(1, n_classes)

    # Create a function that filters the predictions for the given batch item. Specifically, it performs:
    # - confidence thresholding
    # - non-maximum suppression (NMS)
    # - top-k filtering
    def filter_predictions(batch_item):

        # Create a function that filters the predictions for one single class.
        def filter_single_class(index):

            # From a tensor of shape (n_boxes, n_classes + 4 coordinates) extract
            # a tensor of shape (n_boxes, 1 + 4 coordinates) that contains the
            # confidnece values for just one class, determined by `index`.
            confidences = tf.expand_dims(batch_item[..., index], axis=-1)
            class_id = tf.fill(dims=tf.shape(confidences), value=float(index))
            box_coordinates = batch_item[..., -4:]

            single_class = tf.concat([class_id, confidences, box_coordinates], axis=-1)

            # Apply confidence thresholding with respect to the class defined by `index`.
            threshold_met = single_class[:, 1] > tf_confidence_thresh
            single_class = tf.boolean_mask(tensor=single_class,
                                           mask=threshold_met)

            # If any boxes made the threshold, perform NMS.
            def perform_nms():
                scores = single_class[..., 1]

                # `tf.image.non_max_suppression()` needs the box coordinates in the format `(ymin, xmin, ymax, xmax)`.
                xmin = tf.expand_dims(single_class[..., -4], axis=-1)
                ymin = tf.expand_dims(single_class[..., -3], axis=-1)
                xmax = tf.expand_dims(single_class[..., -2], axis=-1)
                ymax = tf.expand_dims(single_class[..., -1], axis=-1)
                boxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=-1)

                maxima_indices = tf.image.non_max_suppression(boxes=boxes,
                                                              scores=scores,
                                                              max_output_size=tf_nms_max_output_size,
                                                              iou_threshold=iou_threshold,
                                                              name='non_maximum_suppresion')
                maxima = tf.gather(params=single_class,
                                   indices=maxima_indices,
                                   axis=0)
                return maxima

            def no_confident_predictions():
                return tf.constant(value=0.0, shape=(1, 6))

            single_class_nms = tf.cond(tf.equal(tf.size(single_class), 0), no_confident_predictions, perform_nms)

            # Make sure `single_class` is exactly `self.nms_max_output_size` elements long.
            padded_single_class = tf.pad(tensor=single_class_nms,
                                         paddings=[[0, tf_nms_max_output_size - tf.shape(single_class_nms)[0]], [0, 0]],
                                         mode='CONSTANT',
                                         constant_values=0.0)

            return padded_single_class

        # Iterate `filter_single_class()` over all class indices.
        filtered_single_classes = tf.map_fn(fn=lambda i: filter_single_class(i),
                                            elems=tf.range(1, n_classes),
                                            dtype=tf.float32,
                                            parallel_iterations=128,
                                            back_prop=False,
                                            swap_memory=False,
                                            infer_shape=True,
                                            name='loop_over_classes')

        # Concatenate the filtered results for all individual classes to one tensor.
        filtered_predictions = tf.reshape(tensor=filtered_single_classes, shape=(-1, 6))

        # Perform top-k filtering for this batch item or pad it in case there are
        # fewer than `self.top_k` boxes left at this point. Either way, produce a
        # tensor of length `self.top_k`. By the time we return the final results tensor
        # for the whole batch, all batch items must have the same number of predicted
        # boxes so that the tensor dimensions are homogenous. If fewer than `self.top_k`
        # predictions are left after the filtering process above, we pad the missing
        # predictions with zeros as dummy entries.
        def top_k():
            return tf.gather(params=filtered_predictions,
                             indices=tf.nn.top_k(filtered_predictions[:, 1], k=tf_top_k, sorted=True).indices,
                             axis=0)

        def pad_and_top_k():
            padded_predictions = tf.pad(tensor=filtered_predictions,
                                        paddings=[[0, tf_top_k - tf.shape(filtered_predictions)[0]], [0, 0]],
                                        mode='CONSTANT',
                                        constant_values=0.0)
            return tf.gather(params=padded_predictions,
                             indices=tf.nn.top_k(padded_predictions[:, 1], k=tf_top_k, sorted=True).indices,
                             axis=0)

        top_k_boxes = tf.cond(tf.greater_equal(tf.shape(filtered_predictions)[0], tf_top_k), top_k, pad_and_top_k)

        return top_k_boxes

    # Iterate `filter_predictions()` over all batch items.
    output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
                              elems=y_pred,
                              dtype=None,
                              parallel_iterations=128,
                              back_prop=False,
                              swap_memory=False,
                              infer_shape=True,
                              name='loop_over_batch')

    return output_tensor


out = decode(y_pred)

for i in out[0]:
    if i[1] < 1 and i[1] > 0.8:
        cv2.putText(
            t_image,
            label_maps[int(i[0])],
            (int(i[2]), int(i[3])),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 0, 0),
            1,
            1
        )
        cv2.rectangle(
            t_image,
            (int(i[2]), int(i[3])),
            (int(i[4]), int(i[5])),
            (0, 255, 0),
            2
        )

cv2.imshow("image", t_image)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
