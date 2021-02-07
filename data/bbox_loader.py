from data import detection_collate
import torch
from torch.utils.data import DataLoader
import random


# build a data loader from a normal VOC like dataset.
# iterable class. each time get batch_size number of bounding boxes of interested labels.
# return list of data like [[xmin, ymin, xmax, ymax], ... ]
class BoundingBoxesLoader:

    def __init__(self, dataset, interest, batch_size=32, shuffle=True, drop_last=False):
        self.bb_data = []
        loader = DataLoader(dataset, 32,
                            num_workers=4,
                            collate_fn=detection_collate,
                            pin_memory=True)
        # keep only bounding boxes of the interested labels
        for data in loader:
            targets_batch = data[1]
            for targets in targets_batch:
                self.bb_data.extend([o[:4] for o in targets if o[4] in interest])
        self.bb_data = torch.tensor(self.bb_data)
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
            return torch.tensor(self.lst[self.ids[self.ptr - self.size: self.ptr]])

    def __iter__(self):
        return self.IterInstance(self.bb_data, self.b_size, self.shuffle, self.drop_last)



