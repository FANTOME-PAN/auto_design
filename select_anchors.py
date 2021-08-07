import argparse
from data import BaseTransform
from data.coco import COCOAnnotationTransform, COCODetection, COCO_ROOT, COCO18_CLASSES
from data.config import voc, coco18, helmet, generic, coco
from data.helmet import HelmetDetection, HELMET_CLASSES, HELMET_ROOT
from data.voc0712 import VOCDetection, VOC_CLASSES, VOC_ROOT
import torch
import os
from utils.anchor_utils import gen_priors, AnchorsPool, trim


parser = argparse.ArgumentParser(
    description='Adaptive prior boxes')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'COCO18', 'helmet'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=None,
                    help='Dataset root directory path')
parser.add_argument('--batch_size', default=256, type=int,
                    help='Batch size for training')
parser.add_argument('--save_pth', default='selected_priors.pth',
                    help='cache for truths of given dataset')
parser.add_argument('--priors_pth', default='params.pth',
                    help='path of saved priors')
parser.add_argument('--cache_pth', default='gts_VOC.pth',
                    help='path of cached ground truths')
parser.add_argument('--thresh_types', default=32, type=int,
                    help='threshold for the number of prior box types.')
parser.add_argument('--thresh_boxes', default=8732, type=int,
                    help='threshold for the number of prior boxes.')
parser.add_argument('--gpus', default='1',
                    type=str, help='visible devices for CUDA')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

dataset = None
if args.dataset == 'VOC':
    config = voc
    rt = VOC_ROOT if args.dataset_root is None else args.dataset_root
    dataset = VOCDetection(rt, transform=BaseTransform(300, (104, 117, 123)))
elif args.dataset == 'COCO18':
    config = coco18
    rt = COCO_ROOT if args.dataset_root is None else args.dataset_root
    dataset = COCODetection(rt, transform=BaseTransform(300, (104, 117, 123)),
                            target_transform=COCOAnnotationTransform('COCO18'))
elif args.dataset == 'helmet':
    config = helmet
    rt = HELMET_ROOT if args.dataset_root is None else args.dataset_root
    dataset = HelmetDetection(rt, transform=BaseTransform(300, (104, 117, 123)))
else:
    raise NotImplementedError()
config = generic


if __name__ == '__main__':
    params = torch.load(args.priors_pth)
    prior_types = trim(params)
    pool = AnchorsPool(dataset, prior_types, config, args.thresh_types, args.thresh_boxes, gts_cache=args.cache_pth)

    sig = 0
    while not sig:
        sig = pool.pop()
    if sig == 1:
        priors = [o.cpu() for o in pool.selected_prior_types()]
        # works well
        torch.save(priors, args.save_pth)
        tpl = [o.size(0) for o in priors]
        print('types per layer: ' + str(tpl))
        print('total number of types: ' + str(sum(tpl)))
        print('total number of prior boxes: ' + str(pool.num_selected_boxes))
        for i, o in enumerate(priors):
            print('Layer %d:\n%s' % (i, str(o)))
