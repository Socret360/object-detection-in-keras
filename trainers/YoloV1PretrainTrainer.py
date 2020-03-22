from base.base_trainer import BaseTrainer


class YoloV1PretrainTrainer(BaseTrainer):
  def __init__(self, model, data, config, valid_data):
    super(YoloV1PretrainTrainer, self).__init__(model, data, config, valid_data)

  def train(self):
    self.model.fit(
        x=self.data,
        epochs=self.config.training.epochs,
        validation_data=self.valid_data
    )
