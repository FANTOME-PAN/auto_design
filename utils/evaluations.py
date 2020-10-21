import sys
import numpy as np
import torch
from torch import nn
from data.helmet import HELMET_CLASSES
from data.config import voc
from layers.box_utils import jaccard
import pickle
import cv2
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


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


# detection size: 1 * num_clasess * top_k * 5(score, bbox)
def decode_raw_detection(detection, h, w):
    dets = [torch.tensor([]) for _ in range(detection.size(1) - 1)]
    for cls_idx in range(detection.size(1) - 1):
        cls_det = detection[0, cls_idx + 1]
        mask = (cls_det[:, 0] > 0.).unsqueeze(-1).expand_as(cls_det)
        cls_det = cls_det[mask].view(-1, 5)

        if cls_det.size(0) == 0:
            continue
        assert isinstance(cls_det, torch.Tensor)
        cls_det.clamp_(0., 1.)
        cls_det[:, 1] *= w
        cls_det[:, 3] *= w
        cls_det[:, 2] *= h
        cls_det[:, 4] *= h
        dets[cls_idx] = cls_det
    return dets


# step 1: find gt bbox with the same class and with max IOU for each detection bbox
# step 2: score = IOU * score
# step 3: ret = scores.mean()
def get_conf_gt(detection, h, w, annopath, classes=HELMET_CLASSES, cls_to_ind=None):
    num_classes = len(HELMET_CLASSES)
    dets = decode_raw_detection(detection, h, w)
    assert num_classes == len(dets)
    if cls_to_ind is None:
        cls_to_ind = dict(zip(classes, range(len(classes))))
    rec = parse_rec(annopath)
    bbgt = [torch.tensor([]) for _ in range(num_classes)]
    for cls_idx in range(num_classes):
        bbgt[cls_idx] = torch.tensor([x['bbox'] for x in rec if cls_to_ind[x['name']] == cls_idx], dtype=torch.float)
    bbdet = [dets[i][:, 1:] if dets[i].size(0) > 0 else torch.tensor([]) for i in range(len(dets))]

    cls_ious = [torch.tensor([]) for _ in range(num_classes)]
    for cls_idx in range(num_classes):
        # K * 4
        bb = bbdet[cls_idx]
        # N * 4
        gt = bbgt[cls_idx]
        if gt.size(0) == 0 or bb.size(0) == 0:
            continue
        iou = jaccard(gt, bb).t()
        cls_ious[cls_idx] = iou
    max_ious = [x.max(1)[0] if x.size(0) > 0 else None for x in cls_ious]
    return cls_ious, max_ious


def output_detection_result(img_path, ids, detections, h, w, classes=HELMET_CLASSES,
                            score_thresh=0.7, out_dir='./eval/imgs', annopath=None, show=False):
    dets = decode_raw_detection(detections, h, w)
    img = cv2.imread(img_path % ids)
    color_red = (0, 0, 255)
    color_green = (0, 255, 0)

    if annopath is not None:
        rec = parse_rec(annopath % ids)
        bbgt = [obj['bbox'] for obj in rec]
        for xx1, yy1, xx2, yy2 in bbgt:
            cv2.rectangle(img, (xx1, yy1), (xx2, yy2), color_green, thickness=2)

    for cls_idx in range(len(classes)):
        det = dets[cls_idx]
        for i in range(det.size(0)):
            if det[i, 0] <= score_thresh:
                continue
            xx1, yy1, xx2, yy2 = det[i, 1:].type(torch.int)
            cv2.rectangle(img, (xx1, yy1), (xx2, yy2), color_red, thickness=1)
            cv2.putText(img, text='%s:%.2f' % (classes[cls_idx], det[i, 0].item()),
                        org=(xx1 + 2, yy1 + 11), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.4, color=color_red)
    if show:
        cv2.imshow(ids[1], img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    path = out_dir + '/%s_det_result.jpg' % ids[1]
    cv2.imwrite(path, img)


# assume that ssd_output is from the output of the SSD net in 'train' phase
# returns batch_num * num_layers * num_classes
# ON CPU
def get_Tk_list(ssd_output, cfg=voc):
    num_layers = len(cfg['feature_maps'])
    batch_num = ssd_output[0].size(0)
    # output matrix
    res = torch.zeros((batch_num, num_layers, cfg['num_classes']), dtype=torch.float).cpu()
    # split output by layers
    splt_points = [0 for _ in range(num_layers + 1)]
    for i, map_len, num_boxes in zip(range(num_layers), cfg['feature_maps'], cfg['num_prior_boxes']):
        splt_points[i + 1] = splt_points[i] + map_len * map_len * num_boxes

    conf_output = nn.Softmax(dim=-1)(ssd_output[1]).cpu()
    for i in range(batch_num):
        splt = [conf_output[i][splt_points[j]:splt_points[j + 1]] for j in range(num_layers)]
        for j in range(num_layers):
            bgs = splt[j][:, 0].view(-1, 1).expand_as(splt[j])
            # exclude the influence from the confs of other classes
            splt[j][:, :] = splt[j] / (splt[j] + bgs)
            # ignore the conf of background
            for k in range(1, cfg['num_classes']):
                # the conf matrix of class k, from layer j, img i in the given batch
                m = splt[j][:, k]
                m = m[m > 0.5]
                if m.size(0) == 0:
                    continue
                # msk = (m <= 0.5).type(torch.float)
                # for each element e in m, e = e if e > 0.5 else 1. - e
                # m += -2. * m * msk + msk
                res[i, j, k] = m.mean().item()

    return res




