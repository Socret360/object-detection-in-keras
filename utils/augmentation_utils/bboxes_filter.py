def bboxes_filter():
    """
    """
    def _augment(
        image,
        bboxes,
        classes=None
    ):
        return image, bboxes, classes
    return _augment
