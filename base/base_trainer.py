class BaseTrainer:
  def __init__(self, model, data, config, valid_data=None):
    self.model = model
    self.data = data
    self.config = config
    self.valid_data = valid_data

  def train(self):
    raise NotImplementedError
