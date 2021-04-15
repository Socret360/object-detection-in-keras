from shapely.geometry import Polygon


def polygon_iou(p1, p2):
    P1 = Polygon(p1)
    P2 = Polygon(p2)
    intersection = P1.intersection(P2).area
    union = P1.area + P2.area - intersection
    res = intersection / union
    return res
