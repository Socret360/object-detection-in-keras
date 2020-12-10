import numpy as np


def one_hot_class_label(classname, label_maps):
    """ Turn classname to one hot encoded label.

    Args:
        - classname: String representing the classname
        - label_maps: A list of strings containing all the classes

    Returns:
        - A numpy array of shape (len(label_maps), )

    Raises:
        - Classname does not includes in label maps
    """
    assert classname in label_maps, "classname must be included in label maps"
    temp = np.zeros((len(label_maps)), dtype=np.int)
    temp[label_maps.index(classname)] = 1
    return temp
