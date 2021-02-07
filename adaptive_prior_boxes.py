import argparse
from data import BaseTransform, detection_collate
from data.bbox_loader import BoundingBoxesLoader
from data.config import voc
from data.voc0712 import VOCAnnotationTransform, VOCDetection, VOC_CLASSES, VOC_ROOT
from layers import PriorBox
from layers.modules.adaptive_prior_boxes_loss import AdaptivePriorBoxesLoss
import torch
from torch import optim


parser = argparse.ArgumentParser(
    description='Adaptive prior boxes')
parser.add_argument('--interest', default='car',
                    type=str, help='the names of labels of interest, split by comma')
parser.add_argument('--beta', default=1., type=float,
                    help='constant that controls the influence of number of prior boxes in the loss function')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'helmet'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--min_batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
args = parser.parse_args()

if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

if args.dataset == 'VOC':
    config = voc
    dataset = VOCDetection(args.dataset_root, [('2007', 'test')],
                           BaseTransform(300, (104, 117, 123)),
                           VOCAnnotationTransform())
    label_dict = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
else:
    raise NotImplementedError()

labels_of_interest = [label_dict[l] for l in args.interest.split(',')]


def train():
    init_boxes = PriorBox(cfg=config).forward()
    # double the number of init boxes
    all_prior_boxes = init_boxes.unsqueeze(1).repeat(1, 2, 1).view(-1, 4)
    locs = all_prior_boxes[:, 0:2]
    params = torch.tensor([[h, w, 0.] for cx, cy, h, w in all_prior_boxes], requires_grad=True)

    # create data loader
    data_loader = BoundingBoxesLoader(dataset, args.batch_size, shuffle=True, drop_last=True)
    b_iter = iter(data_loader)

    # create optimizer
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # create loss function
    loss_fn = AdaptivePriorBoxesLoss(args.beta)

    # train
    for iteration in range(100000):
        try:
            truths = next(b_iter)
        except StopIteration:
            b_iter = iter(data_loader)
            truths = next(b_iter)

        optimizer.zero_grad()
        loss = loss_fn(locs, params, truths)
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print('iter %d: loss=%.4f' % (iteration, loss.item()))


if __name__ == '__main__':
    train()


