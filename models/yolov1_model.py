import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from base import BaseModel


class YoloV1Model(BaseModel):
  def __init__(self, config):
    super(YoloV1Model, self).__init__(config)
    self.build_model()

  def __Conv2D(self, x=None, filters=None, kernel_size=None, strides=(1, 1), name=None):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                               strides=strides, padding="same", use_bias=False, name=name)(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}-bnorm")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1, name=f"{name}-lrelu")(x)
    return x

  def __MaxPooling2D(self, x=None):
    return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

  def build_model(self):
    input_img = tf.keras.layers.Input(
        shape=(
            self.config.training.input_size,
            self.config.training.input_size,
            3
        )
    )
    x = self.__Conv2D(x=input_img, filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv-1")
    x = self.__MaxPooling2D(x=x)
    x = self.__Conv2D(x=x, filters=192, kernel_size=(3, 3), name="conv-2")
    x = self.__MaxPooling2D(x=x)
    #
    x = self.__Conv2D(x=x, filters=128, kernel_size=(1, 1), name="conv-3")
    x = self.__Conv2D(x=x, filters=256, kernel_size=(3, 3), name="conv-4")
    x = self.__Conv2D(x=x, filters=256, kernel_size=(1, 1), name="conv-5")
    x = self.__Conv2D(x=x, filters=512, kernel_size=(3, 3), name="conv-6")
    x = self.__MaxPooling2D(x=x)
    #
    x = self.__Conv2D(x=x, filters=256, kernel_size=(1, 1), name="conv-7")
    x = self.__Conv2D(x=x, filters=512, kernel_size=(3, 3), name="conv-8")
    x = self.__Conv2D(x=x, filters=256, kernel_size=(1, 1), name="conv-9")
    x = self.__Conv2D(x=x, filters=512, kernel_size=(3, 3), name="conv-10")
    x = self.__Conv2D(x=x, filters=256, kernel_size=(1, 1), name="conv-11")
    x = self.__Conv2D(x=x, filters=512, kernel_size=(3, 3), name="conv-12")
    x = self.__Conv2D(x=x, filters=256, kernel_size=(1, 1), name="conv-13")
    x = self.__Conv2D(x=x, filters=512, kernel_size=(3, 3), name="conv-14")
    x = self.__Conv2D(x=x, filters=512, kernel_size=(1, 1), name="conv-15")
    x = self.__Conv2D(x=x, filters=1024, kernel_size=(3, 3), name="conv-16")
    x = self.__MaxPooling2D(x=x)
    #
    x = self.__Conv2D(x=x, filters=512, kernel_size=(1, 1), name="conv-17")
    x = self.__Conv2D(x=x, filters=1024, kernel_size=(3, 3), name="conv-18")
    x = self.__Conv2D(x=x, filters=512, kernel_size=(1, 1), name="conv-19")
    x = self.__Conv2D(x=x, filters=1024, kernel_size=(3, 3), name="conv-20")
    #
    x = self.__Conv2D(x=x, filters=1024, kernel_size=(3, 3), name="conv-21")
    x = self.__Conv2D(x=x, filters=1024, kernel_size=(3, 3), strides=(2, 2), name="conv-22")
    x = self.__Conv2D(x=x, filters=1024, kernel_size=(3, 3), name="conv-23")
    x = self.__Conv2D(x=x, filters=1024, kernel_size=(3, 3), name="conv-24")
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=4096, name='fc-1')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1, name='fc-1-lrelu')(x)
    x = tf.keras.layers.Dropout(rate=0.5, name='fc-1-dropout')(x)
    x = tf.keras.layers.Dense(
        units=(self.config.S * self.config.S * (self.config.B * 5 + self.config.C)),
        activation='linear',
        name='fc-2'
    )(x)
    x = tf.keras.layers.Reshape(
        target_shape=(
            self.config.S,
            self.config.S,
            self.config.B * 5 + self.config.C
        ),
        name='fc-2-reshape'
    )(x)
    model = tf.keras.models.Model(inputs=input_img, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9, decay=5e-4),
        loss=self.__multipart_loss
    )
    return model

  def __xywh_to_x1y1x2y2(self, bbox):
    x = bbox[:, :, :, :, 0]
    y = bbox[:, :, :, :, 1]
    w = bbox[:, :, :, :, 2]
    h = bbox[:, :, :, :, 3]
    half_height = h / 2
    half_width = w / 2
    x1 = x - half_width
    y1 = y - half_height
    x2 = x + half_width
    y2 = y + half_height
    return x1, y1, x2, y2

  def __calc_iou(self, box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    x1 = K.maximum(b1_x1, b2_x1)
    y1 = K.maximum(b1_y1, b2_y1)
    x2 = K.minimum(b1_x2, b2_x2)
    y2 = K.minimum(b1_y2, b2_y2)
    inter_w = K.maximum(0.0, x2 - x1 + 1.0)
    inter_h = K.maximum(0.0, y2 - y1 + 1.0)
    inter_a = (inter_w * inter_h)
    box1_a = (b1_x2 - b1_x1 + 1.0) * (b1_y2 - b1_y1 + 1.0)
    box2_a = (b2_x2 - b2_x1 + 1.0) * (b2_y2 - b2_y1 + 1.0)
    union = (box1_a + box2_a - inter_a)
    return inter_a / union

  def __classes_loss(self, classes_true, classes_pred, obj_mask):
    loss = K.square(classes_true - classes_pred)
    loss = K.sum(loss, axis=[3])
    loss = obj_mask * loss
    loss = K.sum(loss, axis=[1, 2])
    return self.config.lamda_class * loss

  def __confidence_loss(self, c_true, c_pred, obj_mask, noobj_mask):
    l = K.square(c_true - c_pred)
    obj_loss = obj_mask * l
    obj_loss = K.sum(obj_loss, axis=[1, 2, 3])
    noobj_loss = noobj_mask * l
    noobj_loss = K.sum(noobj_loss, axis=[1, 2, 3])
    return self.config.lamda_obj * obj_loss + self.config.lamda_noobj * noobj_loss

  def __coord_loss(self, bboxes_true, bboxes_pred, obj_mask):
    x_true = bboxes_true[:, :, :, :, 0]
    x_pred = bboxes_pred[:, :, :, :, 0]
    y_true = bboxes_true[:, :, :, :, 1]
    y_pred = bboxes_pred[:, :, :, :, 1]
    w_true = bboxes_true[:, :, :, :, 2]
    w_pred = bboxes_pred[:, :, :, :, 2]
    h_true = bboxes_true[:, :, :, :, 3]
    h_pred = bboxes_pred[:, :, :, :, 3]
    loss = K.square(x_true - x_pred)
    loss = loss + K.square(y_true - y_pred)
    loss = loss + K.square(K.sqrt(w_true) - K.sqrt(w_pred))
    loss = loss + K.square(K.sqrt(h_true) - K.sqrt(h_pred))
    loss = obj_mask * loss
    loss = K.sum(loss, axis=[1, 2, 3])
    return self.config.lamda_coord * loss

  def __multipart_loss(self, y_true, y_pred):
    y_pred = K.sigmoid(y_pred)
    #
    bboxes_true = y_true[:, :, :, :self.config.B * 5]
    bboxes_true = K.reshape(bboxes_true, shape=(-1, self.config.S, self.config.S, self.config.B, 5))
    bboxes_pred = y_pred[:, :, :, :self.config.B * 5]
    bboxes_pred = K.reshape(bboxes_pred, shape=(-1, self.config.S, self.config.S, self.config.B, 5))
    #
    bboxes_coords_true = bboxes_true[:, :, :, :, :4]
    bboxes_coords_pred = bboxes_pred[:, :, :, :, :4]
    #
    bboxes_conf_true = bboxes_true[:, :, :, :, 4]
    bboxes_conf_pred = bboxes_pred[:, :, :, :, 4]
    #
    classes_probs_true = y_true[:, :, :, self.config.B * 5:]
    classes_probs_pred = y_pred[:, :, :, self.config.B * 5:]
    #
    iou_scores = self.__calc_iou(
        self.__xywh_to_x1y1x2y2(bboxes_true[:, :, :, :, :4]),
        self.__xywh_to_x1y1x2y2(bboxes_pred[:, :, :, :, :4]),
    )
    #
    boxes_responsible_for_each_grid = K.cast(
        iou_scores >= K.max(iou_scores, -1, keepdims=True),
        dtype=tf.float32
    )
    grids_where_true_bboxes_exist = K.cast(bboxes_conf_true == 1, dtype=tf.float32)
    grids_where_true_bboxes_does_not_exist = K.cast(bboxes_conf_true != 1, dtype=tf.float32)
    obj_mask = boxes_responsible_for_each_grid * grids_where_true_bboxes_exist
    noobj_mask = boxes_responsible_for_each_grid * grids_where_true_bboxes_does_not_exist
    #
    coord_loss = self.__coord_loss(
        bboxes_true[:, :, :, :, :4],
        bboxes_pred[:, :, :, :, :4],
        obj_mask
    )
    confidence_loss = self.__confidence_loss(
        bboxes_conf_true,
        iou_scores,
        obj_mask,
        noobj_mask
    )
    classes_loss = self.__classes_loss(
        classes_probs_true,
        classes_probs_pred,
        grids_where_true_bboxes_exist[:, :, :, 0]
    )
    total_loss = coord_loss + confidence_loss + classes_loss
    total_loss = total_loss / self.config.training.batch_size
    return total_loss
