from utils.evaluations import get_Tk_list
from data import *
from ssd import build_ssd
import torch
from data.voc0712 import VOC_CLASSES
from utils.my_args import Arguments
from torch import nn


def auto_design_tmp(net, dataset, data_loader, cfg=voc):
    # assert ssd_net.phase == 'train'
    batch_iterator = iter(data_loader)
    # num_layers * num_classes
    # ak_lst[layer_idx, class_idx] = score of the class on the layer
    ak_lst = torch.zeros((len(cfg['feature_maps']), cfg['num_classes']), dtype=torch.float).cpu()
    total_imgs = 0.
    while True:
        # load train data
        try:
            images, _ = next(batch_iterator)
        except StopIteration:
            break
        if args.cuda:
            images = images.cuda()
        total_imgs += images.size(0)
        ssd_out = net(images)
        del images
        out = [o.cpu() for o in ssd_out]
        del ssd_out
        tk_lst = get_Tk_list(out, cfg)
        ak_lst += tk_lst.sum(dim=0)
        print(total_imgs)
        if total_imgs >= 4000:
            break
    ak_lst /= total_imgs
    ak_lst = ak_lst.t()[1:]
    ak_lst = ak_lst / (ak_lst.sum(dim=1).view(-1, 1).expand_as(ak_lst))
    for i, cls in enumerate(VOC_CLASSES):
        print('{:>12}:  {}'.format(cls, '  '.join(['%.2f' % o.item() for o in ak_lst[i]])))
    print(ak_lst)


def main():
    if args.cuda:
        args.try_enable_cuda()
    dataset, data_loader = args.init_dataset()
    cfg = args.cfg
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    ssd_net.load_weights(args.resume_pth)
    if args.cuda:
        ssd_net = ssd_net.cuda()
    auto_design_tmp(ssd_net, dataset, data_loader, cfg)


if __name__ == '__main__':
    args = Arguments()
    args.cuda = True
    args.batch_size = 8
    main()

