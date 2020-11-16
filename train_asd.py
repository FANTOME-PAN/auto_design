from data import *
from utils.augmentations import ASDAugmentation
from layers.modules import MultiBoxLoss
from asd import build_asd
from auto_design import get_config_voc
from data.voc0712 import VOC_ROOT, VOCDetection, VOCAnnoTransformForASD, VOC_CLASSES
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'helmet'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=HELMET_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--class_name', default='sheep', choices=VOC_CLASSES,
                    help='the given class name')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        raise NotImplementedError()
    elif args.dataset == 'VOC':
        # cfg = voc
        cfg = get_config_voc(args.class_name, None)
        dataset = VOCDetection(root=VOC_ROOT, transform=ASDAugmentation(cfg['min_dim'], MEANS),
                               target_transform=VOCAnnoTransformForASD(args.class_name))
    else:
        raise RuntimeError()

    asd_net = build_asd('train', cfg=cfg)
    net = asd_net

    if args.cuda:
        net = torch.nn.DataParallel(asd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        asd_net.load_weights(args.resume)
    else:
        base_net_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        asd_net.base.load_state_dict(base_net_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        asd_net.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for step in cfg['lr_steps']:
        if args.start_iter > step:
            print('over %d steps, adjust lr' % step)
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
        else:
            break
    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')
            print('timer: %.4f sec.' % (t1 - t0))

        if iteration != 0 and iteration % 2000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(asd_net.state_dict(), ('weights/cache/asd_net_%s_' % args.dataset) +
                       repr(iteration) + '.pth')
    torch.save(asd_net.state_dict(),
               args.save_folder + 'asd_net_' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    # init.xavier_uniform(param)
    nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight)
        m.bias.zero_()


if __name__ == '__main__':
    train()
