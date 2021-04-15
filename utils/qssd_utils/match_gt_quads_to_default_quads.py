import numpy as np
from .polygon_iou import polygon_iou


def match_gt_quads_to_default_quads(
    gt_quads,
    default_quads,
    match_threshold=0.5,
    neutral_threshold=0.4
):
    assert len(gt_quads.shape) == 3, "Shape of ground truth boxes array must be (n, 4, 2)"
    assert len(default_quads.shape) == 3, "Shape of default boxes array must be (n, 4, 2)"

    num_gt_quads = gt_quads.shape[0]
    num_default_quads = default_quads.shape[0]

    matches = []
    neutral_matches = []

    for i in range(num_gt_quads):
        ious = np.zeros(num_default_quads)
        for j in range(num_default_quads):
            ious[j] = polygon_iou(gt_quads[i], default_quads[j])
        # max matches
        matches.append([i, np.argmax(ious)])
        ious[np.argmax(ious)] = 0
        # threshold matches
        tm = np.argwhere(ious > match_threshold)
        num_tm = tm.shape[0]
        matches += list(np.concatenate([
            np.tile(np.expand_dims(np.array([i]), axis=-1), (num_tm, 1)),
            tm
        ], axis=-1))
        ious[tm] = 0
        # neutral_matches
        nm = np.argwhere(ious > neutral_threshold)
        num_nm = nm.shape[0]
        neutral_matches += list(np.concatenate([
            np.tile(np.expand_dims(np.array([i]), axis=-1), (num_nm, 1)),
            nm
        ], axis=-1))
        ious[nm] = 0

    matches = np.array(matches)
    neutral_matches = np.array(neutral_matches)

    return matches, neutral_matches
