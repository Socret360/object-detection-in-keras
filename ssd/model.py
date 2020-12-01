import cv2
import math
import numpy as np
from models import SSD300_VGG16

# image = cv2.imread("./ssd/test.jpeg")
# image_size = 300
# image = cv2.resize(image, (image_size, image_size))

# scales = np.linspace(0.2, 0.9, 6+1)

# print(scales)
# feature_map_height, feature_map_width = 3, 3
# feature_map_size = min(feature_map_height, feature_map_width)
# offset_x, offset_y = 0.5, 0.5
# grid_size = image_size / feature_map_size
# aspect_ratios = [1.0, 2.0, 0.5]
# scale = scales[0]
# next_scale = scales[0+1]

# # print(scale, next_scale)
# num_default_boxes = len(aspect_ratios) + 1 if 1.0 in aspect_ratios else len(aspect_ratios)

# wh_list = []
# for ar in aspect_ratios:
#     if ar == 1.0:
#         wh_list.append([
#             image_size * np.sqrt(scale * next_scale) * np.sqrt(ar),
#             image_size * np.sqrt(scale * next_scale) * (1 / np.sqrt(ar)),
#         ])
#     wh_list.append([
#         image_size * scale * np.sqrt(ar),
#         image_size * scale * (1 / np.sqrt(ar)),
#     ])
# wh_list = np.array(wh_list, dtype=np.float)

# cx = np.linspace(offset_x * grid_size, image_size - (offset_x * grid_size), feature_map_size)
# cy = np.linspace(offset_y * grid_size, image_size - (offset_y * grid_size), feature_map_size)
# cx_grid, cy_grid = np.meshgrid(cx, cy)
# cx_grid, cy_grid = np.expand_dims(cx_grid, axis=-1), np.expand_dims(cy_grid, axis=-1)
# cx_grid, cy_grid = np.tile(cx_grid, (1, 1, num_default_boxes)), np.tile(cy_grid, (1, 1, num_default_boxes))

# default_boxes = np.zeros((feature_map_height, feature_map_width, num_default_boxes, 4))

# default_boxes[:, :, :, 0] = cx_grid
# default_boxes[:, :, :, 1] = cy_grid
# default_boxes[:, :, :, 2] = wh_list[:, 0]
# default_boxes[:, :, :, 3] = wh_list[:, 1]

# print(default_boxes[:, :, :, [0, 2]].shape)

# print(default_boxes[:, :, :, 2])
# print(num_default_boxes)
# print(default_boxes.shape, cx_grid.shape)

# for i in range(feature_map_size):
#     for j in range(feature_map_size):
#         cv2.circle(image, (int(cx[i]), int(cy[j])), 1, (255, 0, 0), 1)

# for i in range(0, feature_map_size):
#     for j in range(0, feature_map_size):
#         cx = int((i+0.5) * grid_size)
#         cy = int((j+0.5) * grid_size)
#         cv2.circle(image, (cx, cy), 1, (255, 0, 0), 1)
#         if i == 2 and j == 2:
#             for k, ar in enumerate(aspect_ratios):
#                 default_box_width = input_size * scale * math.sqrt(ar)
#                 default_box_height = input_size * scale * (1 / math.sqrt(ar))
#                 half_width = default_box_width // 2
#                 half_height = default_box_height // 2
#                 top_left = (int(cx - (default_box_width // 2)), int(cy - (default_box_height // 2)))
#                 bottom_right = (int(cx + (default_box_width // 2)), int(cy + (default_box_height // 2)))
#                 if k == 0:
#                     cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2, 1)
#                 elif k == 1:
#                     cv2.rectangle(image, top_left, bottom_right, (255, 0, 255), 2, 1)
#                 elif k == 2:
#                     cv2.rectangle(image, top_left, bottom_right, (0, 255, 255), 2, 1)
# cv2.line(image, ((i+1) * grid_size, 0), ((i+1) * grid_size, 300), (255, 255, 255), 1)
# cv2.line(image, (0, (i+1) * grid_size), (300, (i+1) * grid_size), (255, 255, 255), 1)

# cv2.imshow("image", image)

# if cv2.waitKey(0) == ord('q'):
#     cv2.destroyAllWindows()
model = SSD300_VGG16()
model.summary()
