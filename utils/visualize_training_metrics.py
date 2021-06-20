import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description='Visualize training metrics.')
parser.add_argument('logfile', type=str, help='path to dataset dir.')
args = parser.parse_args()

assert os.path.exists(args.logfile), "logfile does not exist"

data = pd.read_csv(args.logfile)

plt.plot(data["epoch"], data["loss"], label="loss")
plt.plot(data["epoch"], data["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
