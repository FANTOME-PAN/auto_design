import torch
from torch import cuda, nn, optim
from utils.evaluations import get_Tk_list
from ssd import SSD
from data.config import voc


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


