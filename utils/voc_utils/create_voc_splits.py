import os
import argparse
from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create a split files')
    parser.add_argument('--images')
    parser.add_argument('--labels')
    args = parser.parse_args()
    images_glob = args.images
    labels_glob = args.labels
    images = glob(images_glob)
    labels = glob(images_glob)
    print(args.images)
