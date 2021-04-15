def get_number_default_quads(aspect_ratios, angles, extra_box_for_ar_1=True):
    num_aspect_ratios = len(aspect_ratios)
    n = num_aspect_ratios + 1 if (1.0 in aspect_ratios) and extra_box_for_ar_1 else num_aspect_ratios
    return n * len(angles)
