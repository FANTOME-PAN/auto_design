import torch
from torch.utils.data import DataLoader
from data.config import voc, helmet, MEANS
from data.helmet import HELMET_ROOT, HelmetDetection
from data.voc0712 import VOC_ROOT, VOCDetection
from utils.augmentations import SSDAugmentation
from data import detection_collate


class Arguments:
    def __init__(self):
        self.dataset = 'VOC'
        self.cuda = True
        self.resume_pth = 'weights/big_net_VOC.pth'
        self.batch_size = 32
        self.num_workers = 4
        self.cfg = None

    def init_dataset(self, init_dataloader=True):
        if self.dataset == 'COCO':
            raise NotImplementedError()
        elif self.dataset == 'VOC':
            # raise NotImplementedError()
            self.cfg = voc
            dataset = VOCDetection(root=VOC_ROOT, transform=SSDAugmentation(self.cfg['min_dim'], MEANS))
        elif self.dataset == 'helmet':
            self.cfg = helmet
            dataset = HelmetDetection(root=HELMET_ROOT, transform=SSDAugmentation(self.cfg['min_dim'], MEANS))
        else:
            raise RuntimeError()
        if not init_dataloader:
            return dataset
        data_loader = DataLoader(dataset, self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=True, collate_fn=detection_collate,
                                 pin_memory=True)
        return dataset, data_loader

    def try_enable_cuda(self):
        if torch.cuda.is_available():
            if self.cuda:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if not self.cuda:
                print("WARNING: It looks like you have a CUDA device, but aren't " +
                      "using CUDA.\nRun with --cuda for optimal training speed.")
                torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

