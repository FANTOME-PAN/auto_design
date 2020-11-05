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
    return parser.parse_args()


def weights_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            m.bias.zero_()


class Wrapper(nn.Module):
    def __init__(self, net: AutoTailoredSmallDetector, num_classes):
        super(Wrapper, self).__init__()
        self.base = net.vgg
        self.pool = nn.AdaptiveMaxPool2d(2)
        self.extras = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = x
        for layer in self.base:
            out = layer(out)
        out = self.pool(out)
        out = self.extras(out.view(x.size(0), -1))
        return out


# temporary
def make_dataset(name='VOC', ratio=3):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    if name == 'VOC':
        # torch.cuda.set_device(1)
        cfg = voc_sd_sofa
        cls_idx = VOC_CLASSES.index('sofa')
        bsize = 1024
        dataset = VOCDetection(root=VOC_ROOT, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        if not os.path.exists('msk_cache.pth'):
            data_loader = data.DataLoader(dataset, bsize,
                                          num_workers=args.num_workers, collate_fn=detection_collate,
                                          pin_memory=True)
            msk = torch.zeros((len(dataset),), dtype=torch.uint8).cuda()
            for i, batch in enumerate(data_loader):
                gt = batch[1]
                csize = len(gt)
                gt = [1 if cls_idx in ann[:, -1] else 0 for ann in gt]
                gt = torch.tensor(gt, dtype=torch.uint8).cuda()
                gt = gt.nonzero().flatten() + i * bsize
                msk[gt] = 1
                print('%d / %d' % (i * bsize + csize, len(dataset)))
            # for i in range(len(dataset)):
            #     img, gt = dataset[i]
            #     if cls_idx in gt[:, -1]:
            #         msk[i] = 1
            #         print('%d / %d' % (i + 1, len(dataset)))
            torch.save(msk.cpu(), 'msk_cache.pth')
        else:
            msk = torch.load('msk_cache.pth').cuda()
        import random
        total = msk.sum().item()
        max_num = round(min(total * ratio, len(dataset) - total))
        reserved = random.sample((~msk).nonzero().tolist(), max_num)
        msk[reserved] = 1
        f07 = open('trainval_sofa07.txt', 'w')
        f12 = open('trainval_sofa12.txt', 'w')
        for i in range(msk.size(0)):
            if msk[i] == 1:
                if 'VOC2007' in dataset.ids[i][0]:
                    f07.write('%s\n' % dataset.ids[i][1])
                else:
                    f12.write('%s\n' % dataset.ids[i][1])
        f07.close()
        f12.close()
    else:
        raise NotImplementedError()


def pretrain_basenet(asd_net: AutoTailoredSmallDetector, cls='sofa'):
    import os
    if not os.path.exists('weights/cache'):
        os.mkdir('weights/cache')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
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
        dataset = VOCDetection(root=VOC_ROOT, image_sets=[('2007', 'trainval_sofa'), ('2012', 'trainval_sofa')],
                               transform=SSDAugmentation(cfg['min_dim'], MEANS))
    else:
        raise RuntimeError()

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        wrapper_net.load_state_dict(torch.load(args.resume))
    else:
        wrapper_net.apply(weights_init)

    if args.cuda:
        asd_net = asd_net.cuda()
        wrapper_net = wrapper_net.cuda()
        net = torch.nn.DataParallel(wrapper_net)
        cudnn.benchmark = True
    else:
        net = wrapper_net

    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    net.train()

    # loss counters
    print('Loading the dataset...')
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

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
        targets = [1. if cls_idx in ann[:, -1].type(torch.int) else 0. for ann in targets]
        gt = torch.tensor(targets, dtype=torch.long).cuda()
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss = loss_fn(out, gt)
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
            torch.save(wrapper_net.state_dict(), ('weights/cache/vgg9lite_wrapper_%s_' % args.dataset) +
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
    # make_dataset()
    asd_net = build_asd('train', cfg=voc_sd_sofa)
    pretrain_basenet(asd_net)

