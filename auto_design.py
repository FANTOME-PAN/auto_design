from utils.evaluations import get_Tk_list
from data import *
from ssd import build_ssd
import torch
from data.voc0712 import VOC_CLASSES
from utils.my_args import Arguments
from torch import nn
import pickle
from layers.squeeze_net import FireModule


def generate_dets(net, dataset, data_loader):
    batch_iterator = iter(data_loader)
    # num_layers * num_classes
    det_lst = []
    total_imgs = len(dataset.ids)
    cnt = 0
    while True:
        # load train data
        try:
            images, _ = next(batch_iterator)
        except StopIteration:
            break
        if args.cuda:
            images = images.cuda()
        cnt += images.size(0)
        ssd_out = net(images)
        del images
        det_lst.append(ssd_out[1].detach().cpu().numpy())
        print('%d / %d' % (cnt, total_imgs))
        if cnt % 4000 == 0:
            with open('det_output_%d.pkl' % cnt, 'wb') as f:
                pickle.dump(det_lst, f)
            det_lst.clear()
            print('saved')
    with open('det_output_%d.pkl' % cnt, 'wb') as f:
        pickle.dump(det_lst, f)
    det_lst.clear()
    print('saved')


def auto_design_tmp(net, dataset, data_loader, cfg=voc):
    # assert ssd_net.phase == 'train'
    batch_iterator = iter(data_loader)
    # num_layers * num_classes
    # ak_lst[layer_idx, class_idx] = score of the class on the layer
    ak_lst = torch.zeros((len(cfg['feature_maps']), cfg['num_classes']), dtype=torch.float).cpu()
    total_imgs = len(dataset.ids)
    cnt = 0
    while True:
        # load train data
        try:
            images, _ = next(batch_iterator)
        except StopIteration:
            break
        if args.cuda:
            images = images.cuda()
        cnt += images.size(0)
        ssd_out = net(images)
        del images
        out = [o.cpu() for o in ssd_out]
        del ssd_out
        tk_lst = get_Tk_list(out, cfg)
        ak_lst += tk_lst.sum(dim=0)
        print('')
        # if total_imgs >= 4000:
        #     break
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
    generate_dets(ssd_net, dataset, data_loader)
    # auto_design_tmp(ssd_net, dataset, data_loader, cfg)


def get_config_voc(class_name, ssd_net, ratio_3x3=0.5, dataset=None):
    assert class_name == 'sheep'
    # use ssd_net to estimate the relevance between the given class the others,
    # like 'horse' and 'cow' to 'sheep', 'train' and 'truck' to 'car'

    # use dataset to determine appropriate prior boxes for the given task

    # use ratio_3x3 to build suitable squeeze net
    from math import ceil as mc, floor as mf
    p, q = 1 - ratio_3x3, ratio_3x3
    base_net = [
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        FireModule(32, 16, mc(128 * p), mf(128 * q)),
        FireModule(128, 32, mc(256 * p), mf(256 * q)),
        nn.MaxPool2d(2),
        FireModule(256, 32, mc(256 * p), mf(256 * q)),
        FireModule(256, 48, mc(384 * p), mf(384 * q)),
        nn.MaxPool2d(2, 2, 1),
        FireModule(384, 48, mc(384 * p), mf(384 * q)),
        FireModule(384, 64, mc(512 * p), mf(512 * q)),
        nn.MaxPool2d(2),
        FireModule(512, 64, mc(512 * p), mf(512 * q))
    ]
    return base_net, None


if __name__ == '__main__':
    args = Arguments()
    args.cuda = True
    args.batch_size = 16
    main()

