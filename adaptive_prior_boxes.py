import argparse
import cv2
from data import BaseTransform, detection_collate
from data.bbox_loader import BoundingBoxesLoader
from data.config import voc
from data.voc0712 import VOCAnnotationTransform, VOCDetection, VOC_CLASSES, VOC_ROOT
from layers import PriorBox
from layers.box_utils import jaccard, point_form
from layers.functions.prior_box import AdaptivePriorBox
from layers.modules.adaptive_prior_boxes_loss import AdaptivePriorBoxesLoss
from math import sqrt
import os
from tensorboardX import SummaryWriter
import torch
from torch import optim
from utils.adaptive_bbox_utils import gen_priors


def str2bool(s):
    ret = {
        'True': True,
        '1':    True,
        'true': True,
        'T':    True,
        'False': False,
        '0':     False,
        'false': False,
        'F':     False,
    }.setdefault(s, False)
    return ret


parser = argparse.ArgumentParser(
    description='Adaptive prior boxes')
parser.add_argument('--interest', default='car',
                    type=str, help='the names of labels of interest, split by comma')
parser.add_argument('--beta', default=1., type=float,
                    help='constant that controls the influence of number of prior boxes in the loss function')
parser.add_argument('--k', default=2.5, type=float,
                    help='influence of best priors within the loss value')
parser.add_argument('--iou_thresh', default=0.4, type=float,
                    help='threshold of minimum IOU that can be included in the loss function')
parser.add_argument('--times_var', default=2, type=int,
                    help='times of variables to the original parameters from which the priors can be generated.')
parser.add_argument('--random_init', default=0.2, type=float,
                    help='give a random init value for each variables. This parameter can control '
                         'the range of random value, based on the original parameters from config.'
                         'e.g., the range of init value is {k}, if set to 0; [0.8k,1.2k], if set to 0.2.')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'helmet'],
                    type=str, help='VOC or COCO')
parser.add_argument('--mode', default='train', choices=['train', 'test', 'compare'],
                    type=str, help='')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--cuda', default='True', type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--cache_pth', default='bounding_boxes_cache.pth',
                    help='cache for truths of given dataset')
parser.add_argument('--save_pth', default='params.pth',
                    help='save path')
parser.add_argument('--cmp_pth', default=None,
                    help='the path of the target to be compared')
parser.add_argument('--cache_interval', default=1000, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--log', default='True', type=str2bool,
                    help='log output loss')
parser.add_argument('--gpus', default='1',
                    type=str, help='visible devices for CUDA')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
if args.log:
    from datetime import datetime
    writer = SummaryWriter('runs/adaptive_priors_loss/%s/' % datetime.now().strftime("%Y%m%d-%H%M%S"))

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
    # original_boxes = PriorBox(cfg=config).forward()
    # # double the number of init boxes
    # init_boxes = original_boxes.unsqueeze(1).repeat(1, 2, 1).view(-1, 4)
    # locs = all_prior_boxes[:, 0:2]
    # params = torch.tensor([[h, w, 0.] for cx, cy, h, w in all_prior_boxes], requires_grad=True)
    priors_generator = AdaptivePriorBox(config, args.times_var)
    original_priors = []
    for k, (min_size, max_size) in enumerate(zip(config['min_sizes'], config['max_sizes'])):
        tmp = torch.zeros(2 + 2 * len(config['aspect_ratios'][k]), 3)
        tmp[0] = torch.tensor([min_size, min_size, 0.])
        medium_size = sqrt(min_size * max_size)
        tmp[1] = torch.tensor([medium_size, medium_size, 0.])
        for kk, ar in enumerate(config['aspect_ratios'][k]):
            tmp[2 * kk + 2] = torch.tensor([min_size * ar, min_size / ar, 0.])
            tmp[2 * kk + 3] = torch.tensor([min_size / ar, min_size * ar, 0.])
        original_priors += [tmp]
    # torch.save([o.detach().cpu()[:, :2] / config['min_dim'] for o in original_priors], 'params_origin.pth')
    # random init
    params = [t.repeat(args.times_var, 1).clone().detach().requires_grad_(True) for t in original_priors]
    with torch.no_grad():
        for p in params:
            p /= config['min_dim']
            rd_init = 1. + args.random_init * (torch.rand(p.size(0), 2) - 0.5)
            p[:, :2] *= rd_init
            p.clamp_(max=1, min=0)
        priors = priors_generator.forward(params)
        locs = priors[:, :2]
    # alphas = torch.zeros(locs.size(0), requires_grad=True)

    # create data loader
    data_loader = BoundingBoxesLoader(dataset, labels_of_interest, args.batch_size, shuffle=True,
                                      drop_last=True, cache_pth=args.cache_pth)
    b_iter = iter(data_loader)

    # create optimizer
    # optimizer = optim.SGD(params + [alphas], lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # create loss function
    loss_fn = AdaptivePriorBoxesLoss(args.beta, args.k, args.iou_thresh)

    # train
    for iteration in range(30000):
        try:
            truths = next(b_iter)
        except StopIteration:
            b_iter = iter(data_loader)
            truths = next(b_iter)

        optimizer.zero_grad()
        v = priors_generator.fast_forward(params)
        loss = loss_fn(locs, v, truths.float().cuda())
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for p in params:
                p[:, :-1].clamp_(max=1, min=0)
        if args.log:
            writer.add_scalar(args.save_pth, loss.item(), iteration + 1)
        # if (iteration + 1) % args.cache_interval == 0:
        #     means = prior_avg_importance(params, priors_generator, alphas)
        #     with torch.no_grad():
        #         for k, p in enumerate(params):
        #             p[:, -1] = means[k]
        #     if not os.path.exists('./cache/'):
        #         os.mkdir('./cache/')
        #     pth = './cache/%s_iter%d.pth' % (args.save_pth, iteration)
        #     torch.save((params, alphas), pth)
        #     print('save cache to %s ' % pth)
        #     layer_avg = [o.mean().item() for o in means]
        #     print('layer avg importance (normalized): ' +
        #           str([p / sum(layer_avg) for p in layer_avg]))
        #     _, ids = torch.cat(means).sort(descending=True)
        #     tp = [p.clone().detach() for p in params]
        #     for k, p in enumerate(tp):
        #         p[:, -1] = k + 1
        #     tp = torch.cat(tp)[ids]
        #     print('top 50 priors: \n%s' % '\n'.join(['L%d-(%.4f, %.4f)' %
        #                                             (int(o[-1]), o[0].item(), o[1].item()) for o in tp[:50]]))
        if (iteration + 1) % args.cache_interval == 0:
            means = [p[:, -1].clone().detach() for p in params]
            if not os.path.exists('./cache/'):
                os.mkdir('./cache/')
            pth = './cache/%s_iter%d.pth' % (args.save_pth, iteration)
            torch.save(params, pth)
            print('save cache to %s ' % pth)
            layer = [o.mean().item() for o in means]
            print('layer importance (normalized): ' +
                  str([p / sum(layer) for p in layer]))
            _, ids = torch.cat(means).sort(descending=True)
            tp = [p.clone().detach() for p in params]
            for k, p in enumerate(tp):
                p[:, -1] = k + 1
            tp = torch.cat(tp)[ids]
            print('top 50 priors: \n%s' % '\n'.join(['L%d-(%.4f, %.4f)' %
                                                    (int(o[-1]), o[0].item(), o[1].item()) for o in tp[:50]]))

        if iteration % 10 == 0:
            print('iter %d: loss=%.4f' % (iteration, loss.item()))
    # means = prior_avg_importance(params, priors_generator, alphas)
    # with torch.no_grad():
    #     for k, p in enumerate(params):
    #         p[:, -1] = means[k]
    torch.save([p.detach().cpu() for p in params], args.save_pth)


def show_priors(background_pth, locs, params, thresh, name='prior boxes', show=True):
    img = cv2.imread(background_pth)
    img = cv2.resize(img, (800, 800))
    color_red = (0, 0, 255)
    params = params.detach()
    _, idx_lst = params[:, -1].sort(descending=True)
    idx_lst = idx_lst[:thresh]
    priors = torch.cat([locs[idx_lst], params[idx_lst][:, :2]], dim=1)
    priors = point_form(priors)
    priors *= 800.
    for xx1, yy1, xx2, yy2 in priors:
        cv2.rectangle(img, (xx1, yy1), (xx2, yy2), color_red, thickness=1)

    if show:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.imwrite('%s.jpg' % name, img)
    pass


def prior_avg_importance(params, priors_gen, alphas):
    means = []
    for k, (start, end) in enumerate(zip(priors_gen.intervals[:-1], priors_gen.intervals[1:])):
        al = alphas[start:end].clone().detach()
        al = al.view(-1, params[k].size(0))
        means += [al.mean(dim=0)]
    return means


def compare(bbox, other):
    means = []
    for k, (p1, p2) in enumerate(zip(bbox, other)):
        exp_p1 = torch.zeros(p1.size(0), 4)
        exp_p2 = torch.zeros(p2.size(0), 4)
        exp_p1[:, 2:] = p1
        exp_p2[:, 2:] = p2
        overlaps = jaccard(
            point_form(exp_p1),
            point_form(exp_p2)
        )  # size [num_p1, num_p2]
        best_overlap, _ = overlaps.max(1)
        means += [best_overlap.mean().item()]
        print("Layer %d avg overlap = %.4f" % (k, means[-1]))
    print("Mean Avg Overlap = %.4f" % (sum(means) / len(means)))
    pass


if __name__ == '__main__':
    # pth = r'E:\hwhit aiot project\auto_design\data\VOCdevkit\VOC2007\JPEGImages\000241.jpg'
    # init_boxes = PriorBox(cfg=config).forward()
    # # double the number of init boxes
    # all_prior_boxes = init_boxes.unsqueeze(1).repeat(1, 2, 1).view(-1, 4)
    # locs = all_prior_boxes[:, 0:2]
    # params = torch.load('params-3.598.pth')
    # show_lst = [5, 10, 25, 50, 90, 200]
    # for th in show_lst:
    #     show_priors(pth, locs, params, th, '%d prior boxes' % th, False)
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        print('\n'.join(['Layer %d:\n%s' % (i, str(o)) for i, o in enumerate(gen_priors(args.save_pth))]))
    elif args.mode == 'compare':
        print('WITH ORIGINAL BBOX')
        compare(gen_priors(args.save_pth), torch.load('params_origin.pth'))
        print('WITH OTHER')
        compare(gen_priors(args.save_pth), gen_priors(args.cmp_pth))

if args.log:
    writer.close()


