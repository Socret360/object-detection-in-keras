def get_number_default_boxes(aspect_ratios, extra_box_for_ar_1=True):
    return len(aspect_ratios) + 1 if (1.0 in aspect_ratios) and extra_box_for_ar_1 else len(aspect_ratios)
