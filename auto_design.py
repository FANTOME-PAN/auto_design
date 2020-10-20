import torch
from torch import cuda, nn, optim
from utils.evaluations import get_Tk_list
from ssd import SSD
from data.config import voc
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from data.voc0712 import VOC_ROOT, VOCDetection
import os
import sys
import time
import torch
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from utils.my_args import Arguments


def auto_design_tmp(ssd_net: SSD, dataset, data_loader, cfg=voc):
    assert ssd_net.phase == 'train'
    batch_iterator = iter(data_loader)
    # num_layers * num_classes
    ak_lst = torch.zeros((len(cfg['feature_maps']), cfg['num_classes']), dtype=torch.float)
    total_imgs = 0.
    while True:
        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            break
        if cuda.is_available():
            images = images.cuda()
        total_imgs += images.size(0)
        out = ssd_net(images)
        tk_lst = get_Tk_list(out, cfg)
        ak_lst += tk_lst.sum(dim=0)
    ak_lst /= total_imgs


def main():
    args = Arguments()
    args.try_enable_cuda()
    dataset, data_loader = args.init_dataset()
    cfg = args.cfg
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes']).cuda()
    ssd_net.load_weights(args.resume_pth)
    auto_design_tmp(ssd_net, dataset, data_loader, cfg)


if __name__ == '__main__':
    main()

