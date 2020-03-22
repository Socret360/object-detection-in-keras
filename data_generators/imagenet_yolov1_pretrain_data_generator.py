import random

import cv2
import numpy as np
from dotmap import DotMap

from base.base_data_generator import BaseDataGenerator


class ImageNetYoloV1PretrainDataGenerator(BaseDataGenerator):
  def __init__(self, config, samples_type="train"):
    super(ImageNetYoloV1PretrainDataGenerator, self).__init__(config)
    if samples_type == "train":
      self.samples = self.__read_sample_file(config.training.dataset.train_sample_file_path)
    elif samples_type == "valid":
      self.samples = self.__read_sample_file(config.training.dataset.valid_sample_file_path)
    elif samples_type == "test":
      self.samples = self.__read_sample_file(config.training.dataset.test_sample_file_path)

    if self.config.training.shuffle_data_samples:
      random.shuffle(self.samples)

    self.labelmaps = self.__read_labelmaps_file(config.training.dataset.labelmaps_file_path)

  def __read_labelmaps_file(self, labelmaps_file_path):
    with open(labelmaps_file_path, "r") as labelmaps_file:
      labelmaps = [val.strip("\n").lower() for val in labelmaps_file.readlines()]
      return labelmaps

  def __read_sample_file(self, sample_file_path):
    with open(sample_file_path, "r") as sample_file:
      lines = sample_file.readlines()
      samples = []
      for line in lines:
        cols = line.split(" ")
        samples.append(
            DotMap({
                "image_path": cols[0].strip("\n"),
                "class_name": cols[1].strip("\n")
            })
        )
      return samples

  def __get_X(self, image_path):
    image = cv2.imread(f"{self.config.training.dataset.dataset_folder_path}/{image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (self.config.training.input_size, self.config.training.input_size))
    image = image / 255
    return np.array(image, dtype=np.float32)

  def __get_y(self, class_name):
    return self.labelmaps.index(class_name.lower())

  def __len__(self):
    return int(np.ceil(len(self.samples) / float(self.config.training.batch_size)))

  def __getitem__(self, idx):
    batch_items = self.samples[idx *
                               self.config.training.batch_size: (idx + 1) * self.config.training.batch_size]
    batch_X, batch_y = [], []

    for item in batch_items:
      X = self.__get_X(item.image_path)
      y = self.__get_y(item.class_name)
      batch_X.append(X)
      batch_y.append(y)

    batch_X = np.array(batch_X, dtype=np.float32)
    batch_y = np.array(batch_y, dtype=np.float32)

    return batch_X, batch_y
