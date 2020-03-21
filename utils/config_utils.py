import json

from dotmap import DotMap


def read_config_json(json_path):
  with open(json_path) as f:
    data = json.load(f)
    config = DotMap(data)
    return config
