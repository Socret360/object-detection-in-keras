def get_number_default_boxes(aspect_ratios, extra_box_for_ar_1=True):
    """ Get the number of default boxes for each grid cell based on the number of aspect ratios
    and whether to add a extra box for aspect ratio 1

    Args:
    - aspect_ratios: A list containing the different aspect ratios of default boxes.
    - extra_box_for_ar_1: Whether to add a extra box for aspect ratio 1.

    Returns:
    - An integer for the number of default boxes.
    """
    num_aspect_ratios = len(aspect_ratios)
    return num_aspect_ratios + 1 if (1.0 in aspect_ratios) and extra_box_for_ar_1 else num_aspect_ratios
