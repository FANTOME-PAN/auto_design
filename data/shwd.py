from .config import HOME
from data.voc0712 import VOCAnnotationTransform, VOCDetection
import os.path as osp

SHWD_CLASSES = (  # always index 0
    'hat', 'person')

SHWD_ROOT = osp.join(HOME, "data/SHWD/")


class SHWDDetection(VOCDetection):
    def __init__(self, root, image_sets=('train', ), transform=None,
                 target_transform=VOCAnnotationTransform(dict(zip(SHWD_CLASSES, range(len(SHWD_CLASSES))))),
                 dataset_name='SHWD'):
        super().__init__(root, tuple(), transform, target_transform, dataset_name)
        for name in image_sets:
            rootpath = self.root
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))



