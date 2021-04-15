import numpy as np
from utils import textboxes_utils


def encode_quads(y, epsilon=10e-5):
    df_bboxes = textboxes_utils.get_bboxes_from_quads(np.reshape(y[:, -16:-8], (-1, 4, 2)))
    y_encoded = y.copy()
    # encoded_x = (x - dx) / (dw) / sqrt(var(x))
    y_encoded[:, [-24, -22, -20, -18]] = (y[:, [-24, -22, -20, -18]] - y[:, [-16, -14, -12, -10]]) / np.tile(np.expand_dims(df_bboxes[:, 2], axis=-1), (1, 4)) / np.sqrt(y[:, [-8, -6, -4, -2]])
    # encoded_y = (y - dy) / (dh) / sqrt(var(y))
    y_encoded[:, [-23, -21, -19, -17]] = (y[:, [-23, -21, -19, -17]] - y[:, [-15, -13, -11, -9]]) / np.tile(np.expand_dims(df_bboxes[:, 3], axis=-1), (1, 4)) / np.sqrt(y[:, [-7, -5, -3, -1]])
    return y_encoded
