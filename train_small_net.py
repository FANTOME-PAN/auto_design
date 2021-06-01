from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd_small import build_small_ssd, SmallSSD, build_mobilenet_ssd, build_mobilenet_v2_ssd
from ssd import SSD, build_ssd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from data.coco import COCO_ROOT, COCODetection
from data.config import coco18, coco_lite
import torch.utils.data as data
import argparse
# from train_big_ssd import adjust_learning_rate, weights_init
from utils.basic_utils import adjust_learning_rate, weights_init
from data.config import helmet_lite, voc, voc_lite, helmet
from data.voc0712 import VOC_ROOT, VOCDetection
from utils.evaluations import parse_rec, voc_ap, get_conf_gt
import os.path as osp


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
annopath = osp.join('%s', 'Annotations', '%s.xml')

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='helmet', choices=['VOC', 'COCO', 'helmet'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=None,
                    help='Dataset root directory path')
parser.add_argument('--bignet', default='big_net_helmet.pth',
                    help='Big net for knowledge distilling')
parser.add_argument('--distill', default=0.5, type=float,
                    help='from 0 to 1, represents the weight of big net output in the calculation of loss')
parser.add_argument('--basenet', default=None,
                    help='Pretrained base model')
parser.add_argument('--ignore_basenet', default=False, type=str2bool,
                    help='Start training without basenet and pretrain')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--sname', default='det_net', choices=['det_net', 'mobi_ssd', 'mobi_v2_ssd'],
                    help='choose type of small net')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--ptr_lr', default=5e-5, type=float,
                    help='initial learning rate for pretrain')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--half', default=False, type=str2bool,
                    help='train with half')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor' if not args.half else 'torch.cuda.HalfTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def train():
    cfg = config_dict.setdefault((args.dataset, args.sname), None)
    bignet_cfg = config_dict.setdefault((args.dataset, 'ssd300'), None)
    if cfg is None:
        raise RuntimeError("No matched config for model %s on dataset %s" % (args.sname, args.dataset))
    if args.dataset == 'COCO':
        rt = args.dataset_root if args.dataset_root is not None else COCO_ROOT
        dataset = COCODetection(root=rt, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'VOC':
        rt = VOC_ROOT if args.dataset_root is None else args.dataset_root
        dataset = VOCDetection(root=rt, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'VOC-v2':
        rt = VOC_ROOT if args.dataset_root is None else args.dataset_root
        dataset = VOCDetection(root=rt, transform=SSDAugmentation(cfg['min_dim'], MEANS),
                               image_sets=[('2007', 'test'), ('2007', 'trainval'), ('2012', 'train6588')])
    elif args.dataset == 'VOC07':
        rt = VOC_ROOT if args.dataset_root is None else args.dataset_root
        dataset = VOCDetection(root=rt,
                               image_sets=[('2007', 'trainval')],
                               transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'helmet':
        rt = args.dataset_root if args.dataset_root is not None else HELMET_ROOT
        dataset = HelmetDetection(root=rt, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    else:
        raise RuntimeError()
    distill_msk = None
    if args.sname == 'det_net':
        distill_msk = torch.ones(8732, dtype=torch.uint8)
        distill_msk[:38 * 38 * 4] = 0
        ssd_net = build_small_ssd('train', cfg['min_dim'], cfg['num_classes'], cfg)
    elif args.sname == 'mobi_v2_ssd':
        assert args.distill < 1e-7
        ssd_net = build_mobilenet_v2_ssd('train', cfg['min_dim'], cfg['num_classes'], cfg)
    # mobile net ssd
    else:
        ssd_net = build_mobilenet_ssd('train', cfg['min_dim'], cfg['num_classes'], cfg)
    if args.cuda:
        ssd_net = ssd_net.cuda()
    if args.half:
        ssd_net.half()
    big_net = build_ssd('train', bignet_cfg['min_dim'], bignet_cfg['num_classes']).cuda()
    big_net.load_state_dict(torch.load(args.save_folder + args.bignet))

    print('Start training...')
    train_detection_net(ssd_net, big_net, dataset, cfg, args.distill > 1e-5, distill_msk)
    torch.save(ssd_net.state_dict(), args.save_folder + '%s_%s.pth' % (args.sname, args.dataset))


def train_vgg(net: SmallSSD, big_net: SSD, data_loader: data.DataLoader, cfg=helmet_lite):
    print('---- training base net ----')
    # optimizer = optim.SGD(net.parameters(), lr=args.ptr_lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=args.ptr_lr)
    loss_fn = nn.MSELoss()
    net.train()

    # loss counters
    total_loss = 0.
    step_index = 0

    # create batch iterator
    batch_iterator = iter(data_loader)
    ts = time.time()
    for iteration in range(20000):
        if iteration in [15000, 17500, 20000]:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index, args.lr)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()

        t0 = time.time()
        with torch.no_grad():
            y_vgg16 = big_net.vgg_forward(images)
        if args.half:
            images.half()
            y_vgg16.half()
        y_vgg_lite = net.basenet_forward(images)
        # back prop
        optimizer.zero_grad()
        loss = loss_fn(y_vgg_lite, y_vgg16)
        loss.backward()
        optimizer.step()
        t1 = time.time()
        total_loss += loss.item()

        if iteration % 5 == 0:
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % loss.item(), end=' ')
            print('timer: %.4f sec.' % (t1 - t0))

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(),
                       'weights/cache/%s_pretrained_%s_%s.pth' % (args.sname, args.dataset, repr(iteration)))
    torch.save(net.state_dict(),
               args.save_folder + '%s_pretrained_%s.pth' % (args.sname, args.dataset))
    pass


def train_detection_net(ssd_net: SmallSSD, big_net: SSD, dataset: data.Dataset, cfg, distill=True, distill_msk=None):
    net = ssd_net
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True, drop_last=True)
    if distill:
        print('Enable distillation')
    else:
        print('Disable distillation')

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        if args.basenet is None:
            if args.ignore_basenet:
                ssd_net.base.apply(weights_init)
            else:
                print('Training base net')
                train_vgg(ssd_net, big_net, data_loader)
        else:
            init_ssd_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network...')
            ssd_net.load_state_dict(init_ssd_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    print('---- training detection net ----')
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)

    # optimizer only for mobilenet v2 ssdlite.
    # assume that betas[0] and betas[1] in pytorch are beta_1 and beta_2 in TensorFlow respectively.
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda, knowledge_distill=distill, use_half=args.half)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0

    step_index = 0
    for step in cfg['lr_steps']:
        if args.start_iter > step:
            print('over %d steps, adjust lr' % step)
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index, args.lr)
        else:
            break
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index, args.lr)

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
        # backprop
        optimizer.zero_grad()
        if distill:
            with torch.no_grad():
                big_pred = big_net(images)
            if args.half:
                images = images.half()
                big_pred = big_pred.half()
            out = net(images)
            loss_l, loss_c, loss_c_distill, loss_l_distill = criterion(out, targets, big_pred, distill_msk)
            loss = loss_l * (1 - args.distill) + loss_l_distill * args.distill \
                + loss_c * (1 - args.distill) + loss_c_distill * args.distill
        else:
            if args.half:
                images = images.half()
            out = net(images)
            # out = (out[0].float(), out[1].float(), out[2])
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

        if iteration != 0 and iteration != args.start_iter and iteration % 1000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/cache/%s_%s_%s.pth' % (args.sname, args.dataset, repr(iteration)))
    torch.save(ssd_net.state_dict(), args.save_folder + '%s_%s.pth' % (args.sname, args.dataset))


if __name__ == '__main__':
    train()
