import tensorflow as tf


class BaseDataGenerator(tf.keras.utils.Sequence):
  def __init__(self):
    """"""

  def __len__(self):
    raise NotImplementedError

  def __getitem__(self, idx):
    raise NotImplementedError
