import argparse

from utils.config_utils import read_config_json
from models.yolov1_pretrain_model import YoloV1PretrainModel
from trainers.YoloV1PretrainTrainer import YoloV1PretrainTrainer
from data_generators.imagenet_yolov1_pretrain_data_generator import ImageNetYoloV1PretrainDataGenerator

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-V", "--version", help="show program version")
  parser.add_argument("-m", "--model", help="model to train")
  args = parser.parse_args()

  if args.model == "yolov1_pretrain":
    config = read_config_json("configs/yolov1_pretrain_config.json")
    model = YoloV1PretrainModel(config)
    train_data_generator = ImageNetYoloV1PretrainDataGenerator(config, samples_type="train")
    valid_data_generator = ImageNetYoloV1PretrainDataGenerator(config, samples_type="valid")
    trainer = YoloV1PretrainTrainer(model.model, train_data_generator, config, valid_data_generator)
    trainer.train()
