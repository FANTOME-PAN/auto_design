from .config import HOME
from data.voc0712 import VOCAnnotationTransform, VOCDetection
import os.path as osp

BCCD_CLASSES = (  # always index 0
    'rbc', 'wbc', 'platelets')

BCCD_ROOT = 'D:\\dataset pan\\BCCD\\'


class BCCDDetection(VOCDetection):
    def __init__(self, root, image_sets=('train', ), transform=None,
                 target_transform=VOCAnnotationTransform(dict(zip(BCCD_CLASSES, range(len(BCCD_CLASSES))))),
                 dataset_name='BCCD'):
        super().__init__(root, tuple(), transform, target_transform, dataset_name)
        for name in image_sets:
            rootpath = self.root
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))



