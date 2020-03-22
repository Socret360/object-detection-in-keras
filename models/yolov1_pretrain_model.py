import tensorflow as tf

from base.base_model import BaseModel


class YoloV1PretrainModel(BaseModel):
  def __init__(self, config):
    super(YoloV1PretrainModel, self).__init__(config)
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
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=self.config.training.num_classes)(x)
    model = tf.keras.models.Model(inputs=input_img, outputs=x)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy'
    )
    self.model = model
