# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser(".")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'num_prior_boxes': [4, 6, 6, 6, 4, 4],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

voc_sd = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'num_prior_boxes': [4, 6, 6, 6, 4, 4],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

voc_sd_sofa = {
    'num_classes': 21,
    'lr_steps': (40000, 50000, 60000),
    'max_iter': 60000,
    'feature_maps': [10, 5, 3, 1],
    'min_dim': 300,
    'steps': [32, 64, 100, 300],
    'min_sizes': [111, 162, 213, 264],
    'max_sizes': [162, 213, 264, 315],
    'aspect_ratios': [[], [3/4, 3/2, 2, 5/2], [3/4, 3/2, 2, 3], [3/4, 3/2, 2, 2/5, 3]],
    'num_prior_boxes': [2, 6, 6, 7],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
#
# coco = {
#     'num_classes': 201,
#     'lr_steps': (280000, 360000, 400000),
#     'max_iter': 400000,
#     'feature_maps': [38, 19, 10, 5, 3, 1],
#     'min_dim': 300,
#     'steps': [8, 16, 32, 64, 100, 300],
#     'min_sizes': [21, 45, 99, 153, 207, 261],
#     'max_sizes': [45, 99, 153, 207, 261, 315],
#     'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
#     'variance': [0.1, 0.2],
#     'clip': True,
#     'name': 'COCO',
# }

helmet = {
    'num_classes': 5,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'helmet',
}

helmet_lite = {
    'num_classes': 5,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [16, 32, 64, 100, 300],
    'min_sizes': [60, 111, 162, 213, 264],
    'max_sizes': [111, 162, 213, 264, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'helmet',
}

