from data import detection_collate_cuda, BaseTransform
from data.bccd import BCCD_ROOT, BCCDDetection, BCCD_CLASSES
from data.coco import COCO_ROOT, COCODetection,  COCO_CLASSES
from data.shwd import SHWD_ROOT, SHWDDetection, SHWD_CLASSES
from data.voc0712 import VOC_ROOT, VOCDetection, VOC_CLASSES
import os
import random
import torch
from torch.utils.data import DataLoader


# build a data loader from a normal VOC like dataset.
# iterable class. each time get batch_size number of bounding boxes of interested labels.
# return list of data like [[xmin, ymin, xmax, ymax], ... ]
class BoundingBoxesLoader:

    def __init__(self, dataset, interest=None, batch_size=32, shuffle=True, drop_last=False,
                 cache_pth='bounding_boxes_cache.pth'):
        if os.path.exists(cache_pth):
            self.bb_data = torch.load(cache_pth)
        elif cache_pth:
            self.bb_data = []
            loader = DataLoader(dataset, 512,
                                collate_fn=detection_collate_cuda)
            # keep only bounding boxes of the interested labels
            for data in loader:
                targets_batch = data[1]
                for targets in targets_batch:
                    self.bb_data.extend([o[:4] for o in targets if int(o[4].item()) in interest])
            self.bb_data = torch.stack(self.bb_data)
            torch.save(self.bb_data, cache_pth)
            # print('cache file saved.')
        self.shuffle = shuffle
        self.b_size = batch_size
        self.drop_last = drop_last

    class IterInstance:
        def __init__(self, lst, size, shuffle, drop_last):
            self.lst = lst
            self.ptr = 0
            self.size = size
            self.ids = [i for i in range(lst.size(0))]
            if shuffle:
                random.shuffle(self.ids)
            if drop_last:
                self.ids = self.ids[:int(lst.size(0) / size) * size]

        def __next__(self):
            self.ptr += self.size
            if self.ptr > len(self.lst):
                raise StopIteration()
            return self.lst[self.ids[self.ptr - self.size: self.ptr]].clone().detach()

    def __iter__(self):
        return self.IterInstance(self.bb_data, self.b_size, self.shuffle, self.drop_last)


if __name__ == '__main__':
    rt = COCO_ROOT
    data_set = COCODetection(root=rt, image_sets=(('2017', 'val'), ), transform=BaseTransform(300, (104, 117, 123)))
    loader = BoundingBoxesLoader(data_set, [i for i in range(len(VOC_CLASSES))],
                                 cache_pth='../truths/gts_coco_17val.pth')
