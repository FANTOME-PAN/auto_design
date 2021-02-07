import matplotlib.pyplot as plt
from data.voc0712 import VOC_ROOT, VOCDetection, VOC_CLASSES
from data.config import MEANS, voc
from utils.augmentations import SSDAugmentation
import pickle
import math
import torch
import numpy as np
from math import sqrt
from itertools import product

rt = '..\\data\\VOCdevkit'
dataset = VOCDetection(root=rt, transform=SSDAugmentation(voc['min_dim'], MEANS))

voc_config_points = []
config = [
    (30, [1, 2, 1/2]),
    (sqrt(30 * 60), [1]),
    (60, [1, 2, 1/2, 3, 1/3]),
    (sqrt(111 * 60), [1]),
    (111, [1, 2, 1/2, 3, 1/3]),
    (sqrt(111 * 162), [1]),
    (162, [1, 2, 1/2, 3, 1/3]),
    (sqrt(213 * 162), [1]),
    (213, [1, 2, 1/2]),
    (sqrt(213 * 264), [1]),
    (264, [1, 2, 1/2]),
    (sqrt(264 * 315), [1]),
]
for _x, _ylst in config:
    voc_config_points.extend(product([_x], _ylst))


def make_points():
    cls_points = [[] for i in range(20)]
    for i in range(len(dataset.ids)):
        _, gt, h, w = dataset.pull_item(i)
        for xmin, ymin, xmax, ymax, cls_idx in gt:
            area_factor = 1. / math.sqrt((xmax - xmin) * (ymax - ymin))
            # width : height
            ratio = (xmax - xmin) / (ymax - ymin)
            cls_idx = int(round(cls_idx))
            cls_points[round(cls_idx)].append((area_factor, ratio))
        if i % 1000 == 0:
            print(str(i))
    with open('points.pkl', 'wb') as f:
        pickle.dump(cls_points, f)


def scatter():
    with open('points.pkl', 'rb') as f:
        cls_points = pickle.load(f)
    lst = [0, 1, 3, 5, 10, 20, 40]

    def map_x(xx):
        for ii, pre, num in zip(range(len(lst) - 1), lst[:-1], lst[1:]):
            if xx < num:
                return (xx - pre) * (10. / (num - pre)) + ii * 10.
        return (xx - lst[-1]) + (len(lst) - 1) * 10.

    def map_y(yy):
        if yy > 1.:
            return yy - 1.
        return -1. / yy + 1.

    for i, cls in enumerate(VOC_CLASSES):
        x = [map_x(t[0]) for t in cls_points[i]]
        y = [map_y(t[1]) for t in cls_points[i]]
        plt.scatter(x, y, alpha=0.16, s=12)
        # draw voc config points
        x = [map_x(300 / t[0]) for t in voc_config_points]
        y = [map_y(t[1]) for t in voc_config_points]
        plt.scatter(x, y, alpha=0.14, s=900, c='r')
        plt.title(cls)
        plt.xticks([map_x(x) for x in lst], [str(x) for x in lst])
        plt.yticks([-3, -2, -1, 0, 1, 2, 3], ['1/4', '1/3', '1/2', '1', '2', '3', '4'])
        plt.xlim([0, 70])
        plt.ylim([-4, 4])
        plt.xlabel('area_factor')
        plt.ylabel('ratio')
        plt.grid(True)
        # plt.show()
        plt.savefig('scatter_plt_%s.jpg' % cls)
        plt.clf()
        # plt.imsave()


# make_points()
scatter()


