"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import HELMET_ROOT, HelmetAnnotationTransform, HelmetDetection, BaseTransform
from data import HELMET_CLASSES, detection_collate_cuda
from data.adapters import IOAdapterSSD
from data.bccd import BCCD_CLASSES, BCCD_ROOT, BCCDDetection
from data.voc0712 import VOC_CLASSES, VOCDetection, VOCAnnotationTransform, VOC_ROOT
from data.coco import COCO_CLASSES, COCO18_CLASSES, COCODetection, COCOAnnotationTransform, COCO_ROOT
from data.config import config_dict, vococo
from data.shwd import SHWD_ROOT, SHWD_CLASSES, SHWDDetection
from datetime import datetime
from layers.functions.prior_box import AdaptivePriorBox
import torch.utils.data as data
from utils.evaluations import get_conf_gt, output_detection_result, get_detection_result
from utils.anchor_utils import gen_priors, AnchorsGenerator
from utils.basic_utils import parse_rec
from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--dataset', default='helmet', choices=['VOC', 'COCO', 'SHWD', 'helmet'],
                    type=str, help='VOC or COCO')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--dataset_root', default=None,
                    help='Dataset root directory path')
parser.add_argument('--set_type', default=None,
                    help='Name of the test list')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--batch_size', default=256, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
# parser.add_argument('--voc_root', default=VOC_ROOT,
#                     help='Location of VOC root directory')
# parser.add_argument('--test_root', default=HELMET_ROOT,
#                     help='Location of VOC root directory')
parser.add_argument('--write_imgs', default=False, type=str2bool,
                    help='write results')
parser.add_argument('--write_det_results', default=False, type=str2bool,
                    help='only write detection results, no evaluation')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--load_dets', default=False, type=str2bool,
                    help='load existing detections')
parser.add_argument('--save_dets', default=False, type=str2bool,
                    help='generate and save detections')
parser.add_argument('--output_mAP', default=True, type=str2bool,
                    help='calculate and output the results of AP to Console')
parser.add_argument('--custom_priors', default=None,
                    help='custom priors for the model')
parser.add_argument('--prior_types', default=32, type=int,
                    help='number of types of prior boxes. a standard value through which the prior boxes is generated.')
parser.add_argument('--gpus', default='1',
                    type=str, help='visible devices for CUDA')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if args.dataset == 'helmet':
    labelmap = HELMET_CLASSES
    root = args.dataset_root or HELMET_ROOT
    annopath = os.path.join(root, 's2', 'Annotations', '%s.xml')
    imgpath = os.path.join(root, 's2', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(root, 's2', 'ImageSets', 'Main') + '/{:s}.txt'
    devkit_path = root + 'helmet'
    set_type = args.set_type or 'test'
elif args.dataset == 'COCO18':
    labelmap = COCO18_CLASSES
    root = args.dataset_root or COCO_ROOT
    annopath = os.path.join(root, 'coco18', 'Annotations', '%s.xml')
    imgpath = os.path.join(root, 'coco18', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(root, 'coco18', 'ImageSets', 'Main') + '/{:s}.txt'
    devkit_path = root + 'coco18'
elif args.dataset == 'COCO':
    labelmap = COCO_CLASSES
    root = args.dataset_root or COCO_ROOT
    if args.set_type is not None:
        year, set_type = args.set_type.split(',')
    else:
        year, set_type = '2017', 'test-dev'
    sub_dir = 'coco' + year
    annopath = os.path.join(root, sub_dir, 'Annotations', '%s.xml')
    imgpath = os.path.join(root, sub_dir, 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(root, sub_dir, 'ImageSets', 'Main') + '/{:s}.txt'
    devkit_path = root + sub_dir
elif args.dataset == 'VOC':
    labelmap = VOC_CLASSES
    root = args.dataset_root or VOC_ROOT
    annopath = os.path.join(root, 'VOC2007', 'Annotations', '%s.xml')
    imgpath = os.path.join(root, 'VOC2007', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(root, 'VOC2007', 'ImageSets', 'Main') + '/{:s}.txt'
    YEAR = '2007'
    devkit_path = root + 'VOC' + YEAR
    set_type = args.set_type or 'test'
elif args.dataset == 'SHWD':
    labelmap = SHWD_CLASSES
    root = args.dataset_root or SHWD_ROOT
    annopath = os.path.join(root, 'Annotations', '%s.xml')
    imgpath = os.path.join(root, 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(root, 'ImageSets', 'Main') + '/{:s}.txt'
    devkit_path = root
    set_type = args.set_type or 'test'
# elif args.dataset == 'VOC-v2':
#     labelmap = VOC_CLASSES
#     root = VOC_ROOT if args.dataset_root is None else args.dataset_root
#     annopath = os.path.join(root, 'VOC2012', 'Annotations', '%s.xml')
#     imgpath = os.path.join(root, 'VOC2012', 'JPEGImages', '%s.jpg')
#     imgsetpath = os.path.join(root, 'VOC2012', 'ImageSets', 'Main') + '/{:s}.txt'
#     YEAR = '2012'
#     devkit_path = root + 'VOC' + YEAR
#     set_type = 'test4952' if args.set_type is None else args.set_type
#
# elif args.dataset == 'VOC07':
#     labelmap = VOC_CLASSES
#     root = VOC_ROOT if args.dataset_root is None else args.dataset_root
#     annopath = os.path.join(root, 'VOC2007', 'Annotations', '%s.xml')
#     imgpath = os.path.join(root, 'VOC2007', 'JPEGImages', '%s.jpg')
#     imgsetpath = os.path.join(root, 'VOC2007', 'ImageSets', 'Main') + '/{:s}.txt'
#     YEAR = '2007'
#     devkit_path = root + 'VOC' + YEAR
#     set_type = 'test' if args.set_type is None else args.set_type

else:
    raise NotImplementedError()

dataset_mean = (104, 117, 123)


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imgsetpath.format(set_type), cls, cachedir,
            ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    cachefile = os.path.join(cachedir, 'annots.pkl')
    if os.path.exists(cachefile):
        print('removing cache')
        os.remove(cachefile)
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            rec = parse_rec(annopath % (imagename))
            if rec is None:
                print('dirty: ' + imagename)
                continue
            recs[imagename] = rec
        # save
        # print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]
    det_results = []
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')
    res_lst = {}
    difficult_lst = []
    data_loader = data.DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=detection_collate_cuda)
    if not args.load_dets:
        for batch_idx, batch in enumerate(data_loader):
            imgs, targets = batch
            print('detecting %d images...' % imgs.size()[0])
            if args.cuda:
                imgs = imgs.cuda()
            with torch.no_grad():
                batch_detections = net(imgs)
            print('finished')

            for ii in range(imgs.size()[0]):
                _t['im_detect'].tic()
                i = args.batch_size * batch_idx + ii
                detections = batch_detections[ii].unsqueeze(0)
                h, w = dataset.cached_hws[i]
                if args.write_det_results:
                        det_results += get_detection_result(dataset.ids[i][1], detections, h, w,
                                                            labelmap, score_thresh=0.01)
                if args.save_dets:
                    res_lst[dataset.ids[i][1]] = detections[0].cpu()
                    print('%s saved' % dataset.ids[i][1])
                if args.write_imgs:
                    _imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
                    _annopath = os.path.join('%s', 'Annotations', '%s.xml')
                    output_detection_result(_imgpath, dataset.ids[i], detections, h, w, classes=labelmap,
                                            score_thresh=0.5,
                                            out_dir='./eval/%s/bignet_output_thresh50' % args.dataset,
                                            annopath=_annopath, show=False)

                # skip j = 0, because it's the background class
                if args.output_mAP and not args.write_det_results:
                    for j in range(1, detections.size(1)):
                        dets = detections[0, j, :]
                        mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                        dets = torch.masked_select(dets, mask).view(-1, 5)
                        if dets.size(0) == 0:
                            continue
                        boxes = dets[:, 1:]
                        boxes[:, 0] *= w
                        boxes[:, 2] *= w
                        boxes[:, 1] *= h
                        boxes[:, 3] *= h
                        scores = dets[:, 0].cpu().numpy()
                        cls_dets = np.hstack((boxes.cpu().numpy(),
                                              scores[:, np.newaxis])).astype(np.float32,
                                                                             copy=False)
                        all_boxes[j][i] = cls_dets

                detect_time = _t['im_detect'].toc(average=False)
                print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

        # for i in range(num_images):
        #     im, gt, h, w = dataset.pull_item(i)
        #     if not args.write_det_results and gt is None:
        #         print('All difficult objects: %s' % dataset.ids[i][1])
        #         difficult_lst.append('ln %d : %s' % (i + 1, dataset.ids[i][1]))
        #         continue
        #     x = im.unsqueeze(0)
        #     if args.cuda:
        #         x = x.cuda()
        #     _t['im_detect'].tic()
        #     with torch.no_grad():
        #         detections = net(x)
        #
        #     if args.write_det_results:
        #         det_results += get_detection_result(dataset.ids[i][1], detections, h, w, labelmap, score_thresh=0.001)
        #
        #     if args.save_dets:
        #         res_lst[dataset.ids[i][1]] = detections[0].cpu()
        #         print('%s saved' % dataset.ids[i][1])
        #     if args.write_imgs:
        #         _imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        #         _annopath = os.path.join('%s', 'Annotations', '%s.xml')
        #         output_detection_result(_imgpath, dataset.ids[i], detections, h, w, classes=labelmap, score_thresh=0.5,
        #                                 out_dir='./eval/%s/bignet_output_thresh50' % args.dataset,
        #                                 annopath=_annopath, show=False)
        #     detect_time = _t['im_detect'].toc(average=False)
        #
        #     # skip j = 0, because it's the background class
        #     if args.output_mAP and not args.write_det_results:
        #         for j in range(1, detections.size(1)):
        #             dets = detections[0, j, :]
        #             mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
        #             dets = torch.masked_select(dets, mask).view(-1, 5)
        #             if dets.size(0) == 0:
        #                 continue
        #             boxes = dets[:, 1:]
        #             boxes[:, 0] *= w
        #             boxes[:, 2] *= w
        #             boxes[:, 1] *= h
        #             boxes[:, 3] *= h
        #             scores = dets[:, 0].cpu().numpy()
        #             cls_dets = np.hstack((boxes.cpu().numpy(),
        #                                   scores[:, np.newaxis])).astype(np.float32,
        #                                                                  copy=False)
        #             all_boxes[j][i] = cls_dets
        #
        #     print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
        #                                                 num_images, detect_time))
        # print('all diffi lst')
        # print('\n'.join(difficult_lst))
        if args.save_dets:
            torch.save(res_lst, 'ssd_detections_%s.pkl' % args.set_type)
        if args.write_det_results:
            name = datetime.now().strftime('%Y-%m-%d_%H-%M')
            torch.save(det_results, args.save_folder + 'detection_results%s.pth' % name)
            return
        if not args.output_mAP:
            return
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(det_file, 'rb') as f:
            all_boxes = pickle.load(f)
    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)
    # my_evaluation(all_boxes, dataset)


def my_evaluation(box_list, dataset):
    pass


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1  # +1 for background
    if args.custom_priors is not None:
        cfg = config_dict[args.dataset]
        apt = IOAdapterSSD(cfg, 'test')
        apt.load(*torch.load(args.custom_priors))
        custom_priors = apt.fit_output(apt.msks[0])
        print('num_boxes = %d ' % custom_priors.size()[0])
        custom_mbox = None
        if args.cuda:
            custom_priors = custom_priors.cuda()
        net = build_ssd('test', cfg, custom_mbox, custom_priors)
    else:
        cfg = config_dict[args.dataset]
        # from data.config import coco_on_voc
        # cfg = coco_on_voc
        net = build_ssd('test', cfg)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    no_anno = args.write_det_results
    # load data
    if args.dataset == 'helmet':
        dataset = HelmetDetection(root, [('s2', set_type)],
                                  BaseTransform(300, dataset_mean),
                                  HelmetAnnotationTransform())
    elif args.dataset == 'VOC':
        dataset = VOCDetection(root, [('2007', set_type)],
                               BaseTransform(300, dataset_mean),
                               VOCAnnotationTransform(), no_anno=no_anno)
    elif args.dataset == 'SHWD':
        dataset = SHWDDetection(root, (set_type,), BaseTransform(300, dataset_mean))
    # elif args.dataset == 'VOC07':
    #     dataset = VOCDetection(root, [('2007', set_type)],
    #                            BaseTransform(300, dataset_mean),
    #                            VOCAnnotationTransform())
    # elif args.dataset == 'VOC-v2':
    #     dataset = VOCDetection(root, [('2012', set_type)],
    #                            BaseTransform(300, dataset_mean),
    #                            VOCAnnotationTransform())
    elif args.dataset == 'COCO18':
        dataset = COCODetection(root, [('18', set_type)],
                                BaseTransform(300, dataset_mean),
                                COCOAnnotationTransform('COCO18'))
    elif args.dataset == 'COCO':
        dataset = COCODetection(root, [(year, set_type)],
                                BaseTransform(300, dataset_mean), no_anno=no_anno)

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)
