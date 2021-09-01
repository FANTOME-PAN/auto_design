from analysis import predict
import argparse
from utils.basic_utils import get_file_name_from_path
import cv2
from data import BaseTransform
from data.adapters import IOAdapterSSD, IOAdapterYOLOv3
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
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'SHWD'],
                    type=str, help='VOC or COCO')
parser.add_argument('--cuda', default='True', type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--batch_size', default=1024, type=int,
                    help='Batch size for training')
parser.add_argument('--test_pths', default='SSD300,params/params_coco17_test1.pth',
                    help='algo1,pth1;algo2,pth2;...')
parser.add_argument('--gpus', default='0',
                    type=str, help='visible devices for CUDA')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

test_gts = torch.load({
    'VOC': 'truths\\gts_voc07test.pth',
    'COCO': 'truths\\gts_coco_17val.pth',
    'SHWD': 'truths\\gts_shwd_test.pth'
}[args.dataset]).float().cuda()

print('name\tloss\tpower1/3\tgeomean\tmean\trecall\tpower3\tbestgts')
template = '\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f'
bl = torch.load('anchors/ssd_coco_anchors.pth')
bl = bl.cuda()
_, bl_a = predict(bl, test_gts, True)
print('ssdcoco' + template % tuple(bl_a))
bl = torch.load('anchors/voc_baseline.pth')
bl = bl.cuda()
_, bl_a = predict(bl, test_gts, True)
print('ssdvoc' + template % tuple(bl_a))
bl = torch.load('anchors/yolov3_anchors.pth')
bl = bl.cuda()
_, bl_a = predict(bl, test_gts, True)
print('yolo' + template % tuple(bl_a))

test_lst = [tuple(o.split(',')) for o in args.test_pths.split(';')]

for i, (algo, pth) in enumerate(test_lst):
    if algo == 'SSD300':
        cfg = config_dict[args.dataset]
        apt = IOAdapterSSD(cfg, 'test')
        apt.load(*torch.load(pth))
        anchs = apt.fit_output(apt.msks[0])
    else:
        with open(pth, 'r') as f:
            txt = f.readline()
        txt = tuple([float(o) for o in txt.split(',')])
        apt = IOAdapterYOLOv3(txt, 'test')
        anchs = apt.get_gen_fn()()
    if args.cuda:
        anchs = anchs.cuda()
    _, inds = predict(anchs, test_gts, True)
    print('num %d' % i + template % tuple(inds))
