from asd import build_asd, AutoTailoredSmallDetector
import argparse
import torch
from torch import nn, optim
import os
from data.config import voc_sd_sofa, MEANS
import time
from data.voc0712 import VOC_ROOT, VOCDetection, VOC_CLASSES
from utils.augmentations import SSDAugmentation
from torch.backends import cudnn
from data import *
import torch.utils.data as data


def parse_args():
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser(
        description='Auto-tailored Small Detector Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                        type=str, help='VOC, COCO, etc.')
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
    parser.add_argument('--lr', '--learning-rate', default=3e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving checkpoint models')
    return parser.parse_args()


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.zero_()


class Wrapper(nn.Module):
    def __init__(self, net: AutoTailoredSmallDetector, num_classes):
        super(Wrapper, self).__init__()
        self.base = net.vgg
        self.extras = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 512),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.base(x).view(x.size(0), -1)
        out = self.extras(out)
        return out


def pretrain_basenet(asd_net: AutoTailoredSmallDetector, cls='sofa'):
    wrapper_net = Wrapper(asd_net, 2)
    cls_idx = VOC_CLASSES.index(cls)
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    if args.dataset == 'COCO':
        raise NotImplementedError()
    elif args.dataset == 'VOC':
        cfg = voc_sd_sofa
        dataset = VOCDetection(root=VOC_ROOT, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    else:
        raise RuntimeError()

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        wrapper_net.load_state_dict(torch.load(args.resume))
    else:
        wrapper_net.apply(weights_init)

    if args.cuda:
        net = torch.nn.DataParallel(wrapper_net)
        cudnn.benchmark = True
    else:
        net = wrapper_net

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

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
    loss_cnt = 0.
    for iteration in range(args.start_iter, 100000):
        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        targets = []
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loss_cnt += loss.item()

        if iteration != 0 and iteration % 10 == 0:
            print('iter ' + repr(iteration) + ' || Loss: %.4f || Avg: %.4f ||'
                  % (loss.item(), loss_cnt * 0.1), end=' ')
            print('timer: %.4f sec.' % (t1 - t0))
            loss_cnt = 0.

        if iteration != 0 and iteration % 2000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(wrapper_net.state_dict(), ('weights/cache/big_net_%s_' % args.dataset) +
                       repr(iteration) + '.pth')
    torch.save(asd_net.state_dict(),
               args.save_folder + 'vgg9_lite_' + args.dataset + '.pth')


# def adjust_learning_rate(optimizer, gamma, step):
#     """Sets the learning rate to the initial LR decayed by 10 at every
#         specified step
#     """
#     lr = args.lr * (gamma ** step)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


if __name__ == '__main__':
    args = parse_args()
