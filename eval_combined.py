"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data.config import *
from torch.autograd import Variable
from data import HELMET_ROOT, HelmetAnnotationTransform, HelmetDetection, BaseTransform
from data import HELMET_CLASSES
from data.voc0712 import VOC_CLASSES, VOCDetection, VOCAnnotationTransform, VOC_ROOT
import torch.utils.data as data
from utils.evaluations import get_conf_gt, output_detection_result
from ssd_small import build_small_ssd, build_mobilenet_ssd, build_mobilenet_v2_ssd
import sys
import os
import time
import argparse
import numpy as np
import pickle
from data.coco import COCO_CLASSES, COCODetection, COCOAnnotationTransform, COCO_ROOT
from ssd import build_ssd
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Detection Network Evaluation')
parser.add_argument('--dataset', default='helmet', choices=['VOC', 'VOC-v2', 'VOC07', 'COCO', 'helmet'],
                    type=str, help='VOC or COCO')
parser.add_argument('--sname', default='det_net', choices=['det_net', 'mobi_ssd', 'mobi_v2_ssd'])
parser.add_argument('--trained_small_model',
                    default='weights/det_net_VOC.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--trained_big_model',
                    default='weights/big_net_VOC.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--dataset_root', default=None,
                    help='Dataset root directory path')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--half', default=False, type=str2bool,
                    help='Use half-precision')
parser.add_argument('--main_lst', default=None,
                    help='Name of the whole test list')
parser.add_argument('--local_lst', default=None,
                    help='Name of the test list for big net')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--load_boxes', default=False, type=str2bool,
                    help='load existing detections')

args = parser.parse_args()

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

set_type = args.main_lst
if args.dataset == 'helmet':
    labelmap = HELMET_CLASSES
    root = HELMET_ROOT if args.dataset_root is None else args.dataset_root
    annopath = os.path.join(root, 'scenario3-share', 'Annotations', '%s.xml')
    imgpath = os.path.join(root, 'scenario3-share', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(root, 'VOC2007', 'ImageSets',
                              'Main', '{:s}.txt')

elif args.dataset == 'COCO':
    labelmap = COCO_CLASSES
    root = args.dataset_root if args.dataset_root is not None else COCO_ROOT
    annopath = os.path.join(root, 'coco18', 'Annotations', '%s.xml')
    imgpath = os.path.join(root, 'coco18', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(root, 'coco18', 'ImageSets', 'Main') + '/{:s}.txt'
    devkit_path = root + 'coco18'
    # set_type = 'test' if args.set_type is None else args.set_type

elif args.dataset == 'VOC':
    labelmap = VOC_CLASSES
    root = VOC_ROOT if args.dataset_root is None else args.dataset_root
    annopath = os.path.join(root, 'VOC2007', 'Annotations', '%s.xml')
    imgpath = os.path.join(root, 'VOC2007', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(root, 'VOC2007', 'ImageSets', 'Main') + '/{:s}.txt'
    YEAR = '2007'
    devkit_path = root + 'VOC' + YEAR
    # set_type = 'test' if args.main_lst is None else args.main_lst

elif args.dataset == 'VOC-v2':
    labelmap = VOC_CLASSES
    root = VOC_ROOT if args.dataset_root is None else args.dataset_root
    annopath = os.path.join(root, 'VOC2012', 'Annotations', '%s.xml')
    imgpath = os.path.join(root, 'VOC2012', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(root, 'VOC2012', 'ImageSets', 'Main') + '/{:s}.txt'
    YEAR = '2012'
    devkit_path = root + 'VOC' + YEAR
    # set_type = 'test4952' if args.set_type is None else args.set_type

elif args.dataset == 'VOC07':
    labelmap = VOC_CLASSES
    root = VOC_ROOT if args.dataset_root is None else args.dataset_root
    annopath = os.path.join(root, 'VOC2007', 'Annotations', '%s.xml')
    imgpath = os.path.join(root, 'VOC2007', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(root, 'VOC2007', 'ImageSets', 'Main') + '/{:s}.txt'
    YEAR = '2007'
    devkit_path = root + 'VOC' + YEAR
    # set_type = 'test' if args.set_type is None else args.set_type

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


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        tmp = obj.find('truncated')
        obj_struct['truncated'] = int(tmp.text) if tmp is not None else None
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


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
            num_images = len(dataset.ids)
            for im_ind, index in enumerate(dataset.ids):
                # small net
                dets = all_boxes[cls_ind+1][im_ind]
                if dets.shape:
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
                # big net
                dets = all_boxes[cls_ind+1][im_ind + num_images]
                if dets.shape:
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
            recs[imagename] = parse_rec(annopath % (imagename))
            for obj in recs[imagename]:
                # change 'helmet-on' and 'helmet-off' to 'helmet_on' and 'helmet_off'
                obj['name'] = obj['name'].replace('-', '_')
            # if i % 100 == 0:
            #     print('Reading annotation for {:d}/{:d}'.format(
            #        i + 1, len(imagenames)))
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
        R = [obj for obj in recs[imagename] if obj['name'].replace(' ', '') == classname]
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


def test_net(save_folder, nets, cuda, dataset, transform, top_k, local_lst,
             im_size=300, thresh=0.05):
    net, bnet = nets
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[np.ndarray([]) for _ in range(num_images << 1)]
                 for _ in range(len(labelmap)+1)]
    cachefile = 'weights/cache/eval_combined_boxes.pkl'
    output_dir = get_output_dir('ssd300_120000', set_type)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    res_lst = {}
    if not args.load_boxes:
        for i in range(num_images):
            im, gt, h, w = dataset.pull_item(i)
            img_name = dataset.ids[i][1]
            x = im.unsqueeze(0)
            if args.cuda:
                x = x.cuda()
            if args.half:
                x = x.half()
            _t['im_detect'].tic()
            with torch.no_grad():
                detections = net(x) if img_name in local_lst else bnet(x)

            detect_time = _t['im_detect'].toc(average=False)

            # skip j = 0, because it's the background class
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

            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                        num_images, detect_time))
        with open(cachefile, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(cachefile, 'rb') as f:
            all_boxes = pickle.load(f)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)
    # my_evaluation(all_boxes, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1                      # +1 for background
    bnet = build_ssd('test', 300, num_classes)
    if args.sname == 'det_net':
        net = build_small_ssd('test', 300, num_classes)            # initialize SSD
    elif args.sname == 'mobi_v2_ssd':
        net = build_mobilenet_v2_ssd('test', 300, num_classes, config_dict[(args.dataset, args.sname)])
    else:  # mobi_ssd
        net = build_mobilenet_ssd('test', 300, num_classes)
    net.load_state_dict(torch.load(args.trained_small_model))
    net.eval()
    bnet.load_state_dict(torch.load(args.trained_big_model))
    bnet.eval()
    print('Finished loading model!')
    # load data
    if args.dataset == 'helmet':
        dataset = HelmetDetection(root, ('scenario3-share', ),
                                  BaseTransform(300, dataset_mean),
                                  HelmetAnnotationTransform())
    elif args.dataset == 'VOC':
        dataset = VOCDetection(root, [('2007', args.main_lst)],
                               BaseTransform(300, dataset_mean),
                               VOCAnnotationTransform())
    elif args.dataset == 'VOC-v2':
        dataset = VOCDetection(root, [('2012', args.main_lst)],
                               BaseTransform(300, dataset_mean),
                               VOCAnnotationTransform())
    elif args.dataset == 'VOC07':
        dataset = VOCDetection(root, [('2007', args.main_lst)],
                               BaseTransform(300, dataset_mean),
                               VOCAnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(root, [('18', args.main_lst)],
                                BaseTransform(300, dataset_mean),
                                COCOAnnotationTransform())
    else:
        raise NotImplementedError()
    root_pth = dataset.ids[0][0]
    with open(os.path.join(root_pth, 'ImageSets', 'Main', args.local_lst + '.txt'), 'r') as _f:
        _txt = _f.read()
        local_lst = _txt.split()
    if args.cuda:
        net = net.cuda()
        bnet = bnet.cuda()
        cudnn.benchmark = True
    if args.half:
        net.half()
    # evaluation
    test_net(args.save_folder, (net, bnet), args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, local_lst, 300,
             thresh=args.confidence_threshold)
