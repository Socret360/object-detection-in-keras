import time
import os
import sys

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from base.base_trainer import BaseTrainer


class YoloV1PretrainTrainer(BaseTrainer):
  def __init__(self, model, data, config, valid_data):
    super(YoloV1PretrainTrainer, self).__init__(model, data, config, valid_data)
    self.callbacks = self.init_callbacks()

  def init_callbacks(self):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    model_output_folder_path = f"{self.config.training.model_checkpoint_folder_path}/{timestr}"
    model_checkpoint_folder_path = f"{model_output_folder_path}/checkpoints"
    tensorboard_log_dir_path = f"{model_output_folder_path}/logs"

    if not os.path.exists(model_checkpoint_folder_path):
      os.makedirs(model_checkpoint_folder_path)

    if not os.path.exists(tensorboard_log_dir_path):
      os.makedirs(tensorboard_log_dir_path)

    model_json = self.model.to_json()
    with open(f"{model_output_folder_path}/model.json", "w") as json_file:
      json_file.write(model_json)

    model_checkpoint_callback = ModelCheckpoint(
        model_checkpoint_folder_path + "/weights.{epoch:02d}-{val_loss:.2f}.h5",
        verbose=1,
        mode='max',
        save_best_only=True,
        monitor='val_accuracy',
        save_weights_only=True,
    )

    tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir_path)
    return [model_checkpoint_callback, tensorboard_callback]

  def train(self):
    history = self.model.fit(
        x=self.data,
        epochs=self.config.training.epochs,
        validation_data=self.valid_data,
        callbacks=self.callbacks
    )
