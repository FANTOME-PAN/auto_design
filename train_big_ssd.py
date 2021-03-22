import argparse
from data import *
from ssd import build_ssd
from data.voc0712 import VOC_ROOT, VOCDetection, VOC_CLASSES
from data.coco18 import COCO_ROOT, COCODetection
from data.config import coco
from layers.functions.prior_box import AdaptivePriorBox
from layers.modules import MultiBoxLoss
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from utils.augmentations import SSDAugmentation
from utils.adaptive_bbox_utils import gen_priors
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'helmet', 'VOC-v2', 'VOC07'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
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
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--custom_priors', default=None,
                    help='custom priors for the model')
parser.add_argument('--prior_types', default=32, type=int,
                    help='number of types of prior boxes. a standard value through which the prior boxes is generated.')
parser.add_argument('--save_name', default='big_net',
                    help='custom name for the trained model')
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

if args.visdom:
    import visdom
    viz = visdom.Visdom()


def train():
    if args.dataset == 'COCO':
        cfg = coco
        rt = args.dataset_root if args.dataset_root is not None else COCO_ROOT
        dataset = COCODetection(root=rt, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'VOC':
        cfg = voc
        dataset = VOCDetection(root=VOC_ROOT, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'VOC-v2':
        cfg = voc
        dataset = VOCDetection(root=VOC_ROOT,
                               image_sets=[('2007', 'test'), ('2007', 'trainval'), ('2012', 'train6588')],
                               transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'VOC07':
        cfg = voc07
        dataset = VOCDetection(root=VOC_ROOT,
                               image_sets=[('2007', 'trainval')],
                               transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'helmet':
        cfg = helmet
        rt = args.dataset_root if args.dataset_root is not None else HELMET_ROOT
        dataset = HelmetDetection(root=rt, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    else:
        raise RuntimeError()
    if args.custom_priors is not None:
        params = torch.load(args.custom_priors)
        bbox = gen_priors(params, args.prior_types, cfg)
        gen = AdaptivePriorBox(cfg, phase='test')
        custom_priors = gen.forward(bbox)
        custom_mbox = [p.size(0) for p in bbox]
        if args.cuda:
            custom_priors = custom_priors.cuda()
        ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'], custom_mbox, custom_priors)
    else:
        ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
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
                             False, best_prior_weight=2.54, use_gpu=args.cuda)

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

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

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
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

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

        if args.visdom:
            update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                            iter_plot, epoch_plot, 'append')

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


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
