from analysis import predict
import argparse
from utils.basic_utils import get_file_name_from_path
import cv2
from data import BaseTransform
from data.adapters import InputAdapterSSD
from data.bbox_loader import BoundingBoxesLoader
from data.coco import COCOAnnotationTransform, COCODetection, COCO_ROOT, COCO_CLASSES
from data.config import config_dict
from data.helmet import HelmetDetection, HELMET_CLASSES, HELMET_ROOT
from data.voc0712 import VOCDetection, VOC_CLASSES, VOC_ROOT
from utils.anchor_utils import AnchorsGenerator
from utils.box_utils import jaccard, point_form
from layers.functions.prior_box import AdaptivePriorBox
from layers.modules.IOUloss import IOULoss, MixedIOULoss
from math import sqrt
import os
from tensorboardX import SummaryWriter
import torch
from torch import optim
from utils.anchor_utils import gen_priors


def str2bool(s):
    ret = {
        'True': True,
        '1': True,
        'true': True,
        'T': True,
        'False': False,
        '0': False,
        'false': False,
        'F': False,
    }.setdefault(s, False)
    return ret


torch_bool = (torch.ones(1) > 0.).dtype

parser = argparse.ArgumentParser(description='train anchors')
parser.add_argument('--interest', default=None,
                    type=str, help='the names of labels of interest, split by comma')
parser.add_argument('--random_init', default=0.2, type=float,
                    help='give a random init value for each variables. This parameter can control '
                         'the range of random value, based on the original parameters from config.'
                         'e.g., the range of init value is {k}, if set to 0; [0.8k,1.2k], if set to 0.2.')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--algo', default='SSD300', choices=['SSD300', 'YOLOv3'],
                    type=str, help='SSD300 or YOLOv3')
parser.add_argument('--mode', default='train', type=str, help='')
parser.add_argument('--cuda', default='True', type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--batch_size', default=256, type=int,
                    help='Batch size for training')
parser.add_argument('--truths_pth', default='truths/trainval_voc.pth',
                    help='cache for truths of given dataset')
parser.add_argument('--save_pth', default='params/params_voc.pth',
                    help='save path')
parser.add_argument('--cmp_pth', default=None,
                    help='the path of the target to be compared')
parser.add_argument('--test_per_cache', default='False', type=str2bool,
                    help='test the current priors after every caching')
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
parser.add_argument('--gpus', default='0',
                    type=str, help='visible devices for CUDA')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
modes = args.mode.split(',')
file_name = get_file_name_from_path(args.save_pth)

if args.log and 'train' in modes:
    from datetime import datetime

    writer = SummaryWriter('runs/adaptive_priors_loss/%s/' % datetime.now().strftime("%Y%m%d-%H%M%S"))

if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

dataset = None
clamp = True
test_gts = torch.load(r'truths\gts_voc07test.pth').float().cuda()


def train():
    if args.algo == 'SSD300':
        apt = InputAdapterSSD(config_dict[args.dataset], args.random_init)
    # args.algo == 'YOLOv3'
    else:
        raise NotImplementedError()

    # init params
    anchs, anch2fmap, fmap2locs, msks = apt.fit_input()

    # create data loader
    data_loader = BoundingBoxesLoader(dataset, None, args.batch_size, shuffle=True,
                                      drop_last=True, cache_pth=args.truths_pth)
    b_iter = iter(data_loader)

    # create optimizer
    # optimizer = optim.SGD(params + [alphas], lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    optimizer = optim.SGD([anchs], lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # create loss function
    loss_fn = MixedIOULoss()

    gen_fn = AnchorsGenerator(anchs, anch2fmap, fmap2locs)
    step = 0
    # train
    for iteration in range(30000):
        try:
            truths = next(b_iter)
        except StopIteration:
            b_iter = iter(data_loader)
            truths = next(b_iter)

        if iteration in (5000, 10000, 15000, 20000, 25000):
            step += 1
            adjust_learning_rate(optimizer, 0.5, step)
        truths = truths.float().cuda() if args.cuda else truths.float()

        optimizer.zero_grad()
        loss = torch.zeros(8)
        for i, msk in enumerate(msks):
            tmp_anchs = gen_fn(msk)
            loss[i] = loss_fn(tmp_anchs, truths)
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            anchs.clamp_(0., 1.)
        if args.log:
            writer.add_scalar(args.save_pth, loss.item(), iteration + 1)

        if (iteration + 1) % 10 == 0:
            print('iter %d: loss=%.4f' % (iteration + 1, loss.item()))

        if (iteration + 1) % args.cache_interval == 0:
            if not os.path.exists('./cache/'):
                os.mkdir('./cache/')
            pth = './cache/%s_iter%d.pth' % (file_name, iteration + 1)
            torch.save((anchs, anch2fmap, fmap2locs, msks), pth)
            print('save cache to %s ' % pth)
            if args.test_per_cache:
                with torch.no_grad():
                    maps = []
                    for i, msk in enumerate(msks):
                        tmp_anchs = gen_fn(msk)
                        maps.append(predict(tmp_anchs, test_gts, True))
                    print('\n'.join(['%dxAnchs = %.4f [loss:%.2f|power1/3:%.4f|geo mean:%.4f|'
                                     'mean:%.4f|recall:%.4f|power3:%.4f|best gt:%.4f]'
                                     % (i + 1, o, *l) for i, (o, l) in enumerate(maps)]))

    for fmap, locs in fmap2locs.items():
        fmap2locs[fmap] = locs.cpu()
    msks = [msk.cpu() for msk in msks]
    torch.save((anchs.detach().cpu(), anch2fmap, fmap2locs, msks), args.save_pth)


def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def test(path):
    print('\n'.join(['Layer %d:\n%s' % (i, str(o)) for i, o in enumerate(gen_priors(path, args.prior_types))]))


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
    if 'train' in modes:
        print("#################################")
        print("########### TRAIN MODE ##########")
        print("#################################")
        train()
    if 'test' in modes:
        print("#################################")
        print("########### TEST MODE ###########")
        print("#################################")
        test(args.save_pth)
    if 'compare' in modes:
        print("#################################")
        print("########## COMPARE MODE #########")
        print("#################################")
        print('WITH ORIGINAL BBOX')
        compare(gen_priors(args.save_pth), torch.load('params_origin.pth'))
        print('WITH OTHER')
        compare(gen_priors(args.save_pth), gen_priors(args.cmp_pth))

if args.log and args.mode == 'train':
    writer.close()
