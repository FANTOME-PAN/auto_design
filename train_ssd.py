import argparse
from data import *
from ssd import build_ssd
from data.adapters import IOAdapterSSD
from data.voc0712 import VOC_ROOT, VOCDetection, VOC_CLASSES
from data.bccd import BCCD_ROOT, BCCDDetection
from data.coco import COCO_ROOT, COCODetection, COCOAnnotationTransform
from data.shwd import SHWD_ROOT, SHWDDetection
from data.config import coco18, voc, helmet, vococo, coco_on_voc, bccd, shwd
from layers.functions.prior_box import AdaptivePriorBox
from layers.modules import MultiBoxLoss
import os
import sys
import warnings
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from utils.augmentations import SSDAugmentation
from utils.anchor_utils import gen_priors, AnchorsGenerator

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'COCO18', 'helmet', 'BCCD', 'SHWD'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=None,
                    help='Dataset root directory path')
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
parser.add_argument('--k', default=1., type=float,
                    help='weight for the best priors in loss function')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--custom_priors', default=None,
                    help='custom priors for the model')
parser.add_argument('--prior_types', default=32, type=int,
                    help='number of types of prior boxes. a standard value through which the prior boxes is generated.')
parser.add_argument('--save_name', default='big_net',
                    help='custom name for the trained model')
parser.add_argument('--gpus', default='0,2,3',
                    type=str, help='visible devices for CUDA')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


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
    if args.dataset == 'COCO18':
        # cfg = coco18
        cfg = vococo
        rt = args.dataset_root or COCO_ROOT
        dataset = COCODetection(root=rt, transform=SSDAugmentation(cfg['min_dim'], MEANS),
                                target_transform=COCOAnnotationTransform('COCO18'))
    elif args.dataset == 'COCO':
        cfg = coco
        # cfg = vococo
        rt = args.dataset_root or COCO_ROOT
        dataset = COCODetection(root=rt, image_sets=(('2017', 'train'),),
                                transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'VOC':
        cfg = voc
        # cfg = coco_on_voc
        rt = args.dataset_root or VOC_ROOT
        dataset = VOCDetection(root=rt, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'BCCD':
        cfg = bccd
        rt = args.dataset_root or BCCD_ROOT
        dataset = BCCDDetection(root=rt, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'SHWD':
        cfg = shwd
        rt = args.dataset_root or SHWD_ROOT
        dataset = SHWDDetection(root=rt, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'helmet':
        cfg = helmet
        rt = args.dataset_root or HELMET_ROOT
        dataset = HelmetDetection(root=rt, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    else:
        raise RuntimeError()
    if args.custom_priors is not None:
        apt = IOAdapterSSD(cfg, 'test')
        apt.load(*torch.load(args.custom_priors))
        custom_priors = apt.fit_output(apt.msks[0])
        print('num_boxes = %d ' % custom_priors.size()[0])
        custom_mbox = None
        # params = torch.load(args.custom_priors)
        # # bbox = gen_priors(params, args.prior_types, cfg)
        # gen = AdaptivePriorBox(cfg, phase='test')
        # custom_priors = gen.forward(params)
        # custom_mbox = [p.size(0) for p in params]
        if args.cuda:
            custom_priors = custom_priors.cuda()
        ssd_net = build_ssd('train', cfg, custom_mbox, custom_priors)
    else:
        # priors = torch.load('anchors/voc_baseline.pth')
        # if args.cuda:
        #     priors = priors.cuda()
        ssd_net = build_ssd('train', cfg)
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net).cuda()
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, best_prior_weight=args.k, use_gpu=args.cuda)

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
            # targets = targets.cuda()
        # else:
        #
        #     targets = [ann for ann in targets]
        # forward
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
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

        if iteration != 0 and iteration % 2000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), ('weights/cache/%s_%s_' % (args.save_name, args.dataset)) +
                       repr(iteration) + '.pth')
    name = '%s_%s' % (args.save_name, args.dataset)
    torch.save(ssd_net.state_dict(),
               args.save_folder + name + '.pth')


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
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    train()
