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
    'lr_steps': (40000, 50000, 60000),
    'max_iter': 60000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


bccd = {
    'num_classes': 4,
    'lr_steps': (40000, 50000, 60000),
    'max_iter': 60000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'BCCD',
}


shwd = {
    'num_classes': 3,
    'lr_steps': (40000, 50000, 60000),
    'max_iter': 60000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'SHWD',
}


voc07 = {
    'num_classes': 21,
    'lr_steps': (40000, 50000, 60000),
    'max_iter': 60000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


voc_lite = {
    'num_classes': 21,
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
    'name': 'VOC',
}


voc07_mobi = {
    'num_classes': 21,
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
    'name': 'VOC',
}

voc07_lite = {
    'num_classes': 21,
    'lr_steps': (40000, 50000, 60000),
    'max_iter': 60000,
    'feature_maps': [19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [16, 32, 64, 100, 300],
    'min_sizes': [60, 111, 162, 213, 264],
    'max_sizes': [111, 162, 213, 264, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco18 = {
    'num_classes': 19,
    'lr_steps': (160000, 200000, 240000),
    'max_iter': 240000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

coco = {
    'num_classes': 81,
    'lr_steps': (160000, 200000, 240000),
    'max_iter': 240000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

coco_on_voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

vococo = {
    'num_classes': 19,
    'lr_steps': (160000, 200000, 240000),
    'max_iter': 240000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC_as_COCO',
}

voc_mobi_v2 = {
    'num_classes': 21,
    'lr_steps': (40000, 100000, 200000),
    'max_iter': 200000,
    'feature_maps': [19, 10, 5, 3, 1, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    # 'min_sizes': [60, 95, 130, 165, 200, 235],
    # 'max_sizes': [95, 130, 165, 200, 235, 270],
    'min_sizes': [60, 105, 150, 195, 240, 285],
    'max_sizes': [60, 150, 195, 240, 285, 300],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'num_prior_boxes': [3, 6, 6, 6, 6, 6],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


coco_mobi_v2 = {
    'num_classes': 19,
    'lr_steps': (50000, 125000, 250000),
    'max_iter': 250000,
    'feature_maps': [19, 10, 5, 3, 1, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 70, 110, 150, 190, 230],
    'max_sizes': [70, 110, 150, 190, 230, 270],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'num_prior_boxes': [4, 6, 6, 6, 4, 4],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

coco_lite = {
    'num_classes': 19,
    'lr_steps': (160000, 200000, 240000),
    'max_iter': 240000,
    'feature_maps': [19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [16, 32, 64, 100, 300],
    'min_sizes': [45, 99, 153, 207, 261],
    'max_sizes': [99, 153, 207, 261, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

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


generic = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'generic',
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


config_dict = {
    'VOC': voc,
    'COCO': coco
}
#

