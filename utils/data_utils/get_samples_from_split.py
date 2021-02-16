import os


def get_samples_from_split(split_file, images_dir, labels_dir):
    """ Create a list of samples that can be feed to a data generator.

    Args:
        - split_file: Path to the dataset's split file. (e.g. train.txt, val.txt)
        - images_dir: Path to images directory.
        - labels_dir: Path to labels directory.

    Returns:
        - A list of samples. Each sample is a string containing paths to both the image file and its corresponding label file separated by space.

    Raises:
        - split_file does not exist.
        - images_dir is not a directory.
        - labels_dir is not a directory.
    """
    assert os.path.isfile(split_file), "split_file does not exists."
    assert os.path.isdir(images_dir), "images_dir is not a directory."
    assert os.path.isdir(labels_dir), "labels_dir is not a directory."

    samples = []
    with open(split_file, "r") as split_file:
        lines = split_file.readlines()
        for line in lines:
            cols = line.split(" ")
            image_filename = cols[0]
            label_filename = cols[1]
            image_file = os.path.join(images_dir, image_filename)
            label_file = os.path.join(labels_dir, label_filename)
            sample = f"{image_file} {label_file}"
            samples.append(sample)
    return samples
