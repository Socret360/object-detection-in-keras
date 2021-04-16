import numpy as np
from .polygon_iou import polygon_iou


def polygon_nms(y_pred, iou_threshold=0.01):
    nms_quads = []
    num_quads = y_pred.shape[0]

    for i in range(num_quads):
        discard = False
        for j in range(num_quads):
            try:
                iou = polygon_iou(
                    np.reshape(y_pred[i, -16:-8], (4, 2)),
                    np.reshape(y_pred[j, -16:-8], (4, 2))
                )

                if iou >= iou_threshold:
                    score_i = y_pred[i, 1]
                    score_j = y_pred[j, 1]
                    if (score_j > score_i):
                        discard = True
            except:
                print("error shape")
                # discard = True

        if not discard:
            nms_quads.append(y_pred[i])

    return np.array(nms_quads)
