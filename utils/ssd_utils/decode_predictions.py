import tensorflow as tf


def decode_predictions(
    y_pred,
    input_size,
    nms_max_output_size=400,
    confidence_threshold=0.01,
    iou_threshold=0.45,
    num_predictions=10
):
    # decode bounding boxes predictions
    cx = y_pred[..., -12] * y_pred[..., -4] * y_pred[..., -6] + y_pred[..., -8]
    cy = y_pred[..., -11] * y_pred[..., -3] * y_pred[..., -5] + y_pred[..., -7]
    w = tf.exp(y_pred[..., -10] * tf.sqrt(y_pred[..., -2])) * y_pred[..., -6]
    h = tf.exp(y_pred[..., -9] * tf.sqrt(y_pred[..., -1])) * y_pred[..., -5]
    # convert bboxes to corners format (xmin, ymin, xmax, ymax) and scale to fit input size
    xmin = (cx - 0.5 * w) * input_size
    ymin = (cy - 0.5 * h) * input_size
    xmax = (cx + 0.5 * w) * input_size
    ymax = (cy + 0.5 * h) * input_size
    # concat class predictions and bbox predictions together
    y_pred = tf.concat([
        y_pred[..., :-12],
        tf.expand_dims(xmin, axis=-1),
        tf.expand_dims(ymin, axis=-1),
        tf.expand_dims(xmax, axis=-1),
        tf.expand_dims(ymax, axis=-1)], -1)
    #
    batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32
    num_boxes = tf.shape(y_pred)[1]
    num_classes = y_pred.shape[2] - 4
    class_indices = tf.range(1, num_classes)
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
            class_id = tf.fill(dims=tf.shape(confidences),
                               value=tf.cast(index, tf.float32))
            box_coordinates = batch_item[..., -4:]

            single_class = tf.concat(
                [class_id, confidences, box_coordinates], -1)

            # Apply confidence thresholding with respect to the class defined by `index`.
            threshold_met = single_class[:, 1] > confidence_threshold
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
                boxes = tf.concat([ymin, xmin, ymax, xmax], -1)
                maxima_indices = tf.image.non_max_suppression(boxes=boxes,
                                                              scores=scores,
                                                              max_output_size=nms_max_output_size,
                                                              iou_threshold=iou_threshold,
                                                              name='non_maximum_suppresion')
                maxima = tf.gather(params=single_class,
                                   indices=maxima_indices,
                                   axis=0)
                return maxima

            def no_confident_predictions():
                return tf.constant(value=0.0, shape=(1, 6))

            single_class_nms = tf.cond(
                tf.equal(tf.size(single_class), 0), no_confident_predictions, perform_nms)

            # Make sure `single_class` is exactly `self.nms_max_output_size` elements long.
            padded_single_class = tf.pad(tensor=single_class_nms,
                                         paddings=[
                                             [0, nms_max_output_size - tf.shape(single_class_nms)[0]], [0, 0]],
                                         mode='CONSTANT',
                                         constant_values=0.0)

            return padded_single_class

        # Iterate `filter_single_class()` over all class indices.
        filtered_single_classes = tf.nest.map_structure(
            tf.stop_gradient,
            tf.map_fn(fn=lambda i: filter_single_class(i),
                      elems=tf.range(1, num_classes),
                      parallel_iterations=128,
                      swap_memory=False,
                      fn_output_signature=tf.TensorSpec(
                          (None, 6), dtype=tf.float32),
                      name='loop_over_classes'))

        # Concatenate the filtered results for all individual classes to one tensor.
        filtered_predictions = tf.reshape(
            tensor=filtered_single_classes, shape=(-1, 6))

        # Perform top-k filtering for this batch item or pad it in case there are
        # fewer than `self.top_k` boxes left at this point. Either way, produce a
        # tensor of length `self.top_k`. By the time we return the final results tensor
        # for the whole batch, all batch items must have the same number of predicted
        # boxes so that the tensor dimensions are homogenous. If fewer than `self.top_k`
        # predictions are left after the filtering process above, we pad the missing
        # predictions with zeros as dummy entries.
        def top_k():
            return tf.gather(params=filtered_predictions,
                             indices=tf.nn.top_k(
                                 filtered_predictions[:, 1], k=num_predictions, sorted=True).indices,
                             axis=0)

        def pad_and_top_k():
            padded_predictions = tf.pad(tensor=filtered_predictions,
                                        paddings=[
                                            [0, num_predictions - tf.shape(filtered_predictions)[0]], [0, 0]],
                                        mode='CONSTANT',
                                        constant_values=0.0)
            return tf.gather(params=padded_predictions,
                             indices=tf.nn.top_k(
                                 padded_predictions[:, 1], k=num_predictions, sorted=True).indices,
                             axis=0)

        top_k_boxes = tf.cond(tf.greater_equal(tf.shape(filtered_predictions)[
                              0], num_predictions), top_k, pad_and_top_k)

        return top_k_boxes

    # Iterate `filter_predictions()` over all batch items.
    output_tensor = tf.nest.map_structure(
        tf.stop_gradient,
        tf.map_fn(fn=lambda x: filter_predictions(x),
                  elems=y_pred,
                  parallel_iterations=128,
                  swap_memory=False,
                  fn_output_signature=tf.TensorSpec(
                      (num_predictions, 6), dtype=tf.float32),
                  name='loop_over_batch'))
    return output_tensor
