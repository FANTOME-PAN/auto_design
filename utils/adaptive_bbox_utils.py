from data.config import voc
import os
import random
import torch
from torch.utils.data import Dataset
from data.coco import COCODetection
from data.voc0712 import VOCDetection
from layers.box_utils import encode, jaccard, point_form
from layers.functions.prior_box import AdaptivePriorBox
from utils.basic_utils import parse_rec

if torch.__version__ < '1.6.0':
    torch_bool = torch.uint8
else:
    torch_bool = torch.bool


def trim(params, iou_thresh=0.8):
    def iou(b1, b2):
        h1, w1 = b1
        h2, w2 = b2
        intersection = min(h1, h2) * min(w1, w2)
        return intersection / (h1 * w1 + h2 * w2 - intersection)

    res = []
    for p in params:
        p = p[p[:, -1].sort(descending=True)[1]]
        msk = torch.ones(p.size(0), dtype=torch_bool)
        for i in range(p.size(0)):
            if msk[i] == 0:
                continue
            # keep p[i], trim all the similar priors
            for j in range(i + 1, p.size(0)):
                if iou(p[i][:2], p[j][:2]) > iou_thresh:
                    msk[j] = 0
        res += [p[msk]]
    return res


# def gen_priors(params, num_types=32, cfg=voc):
#     params = trim(params, iou_thresh=0.75)
#     bbox = [p[:, :2] for k, p in enumerate(params)]
#     return bbox


def gen_priors(params, num_types=32, cfg=voc):
    if isinstance(params, str):
        params = torch.load(params)
    params = trim(params, iou_thresh=0.75)
    means = [p[:, -1].mean().item() for p in params]
    weights = torch.tensor(means).softmax(dim=0).tolist()
    # weights = [m / sum(means) for m in means]
    nums = [min(int(round(w * num_types)), params[k].size(0)) for k, w in enumerate(weights)]
    print('weights by layer: { %s }' % ', '.join(['%.3f' % w for w in weights]))
    print('types by layer: { %s }' % ', '.join(['%d' % n for n in nums]))
    print('%d types of priors in total' % sum(nums))
    print('%d priors in total' % (sum([nums[k] * n * n for k, n in enumerate(cfg['feature_maps'])])))
    # sort by alpha for every layer
    # params = [p[p[:, -1].sort(detscending=True)[1]] for p in params]
    # keep top-k priors in each layer
    bbox = [p[:nums[k], :2] for k, p in enumerate(params)]
    return bbox


class PriorsPool:
    @staticmethod
    # sample the given number of annotations from the given dataset
    def __sample(dataset: (VOCDetection, COCODetection), num_samples):
        anno_pth = dataset._annopath
        anno_ids = random.sample(dataset.ids, min(num_samples, len(dataset.ids)))
        return [anno_pth % o for o in anno_ids]

    @staticmethod
    # generate prior boxes for each type respectively
    def __gen_priors(params, cfg, ret_point_form=True):
        image_size = cfg['min_dim']
        feature_maps = cfg['feature_maps']
        steps = cfg['steps']
        mean = []
        from itertools import product
        for k, f in enumerate(feature_maps):
            mean += [[[] for _ in range(params[k].size(0))]]
            for i, j in product(range(f), repeat=2):
                f_k = image_size / steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                for ii, p in enumerate(params[k]):
                    tmp = torch.zeros(4)
                    tmp[0], tmp[1] = cx, cy
                    tmp[2:] = p[:2]
                    mean[k][ii] += [tmp]
        # back to torch land
        # output = [torch.stack(o).clamp_(max=1, min=0) for o in mean]
        if ret_point_form:
            return [[point_form(torch.stack(boxes).clamp_(max=1, min=0)) for boxes in layer] for layer in mean]
        return [[torch.stack(boxes).clamp_(max=1, min=0) for boxes in layer] for layer in mean]

    def __mk_iou_tensor(self, prior_boxes):
        if isinstance(prior_boxes, int):
            i, j = prior_boxes >> 16, prior_boxes & 0xffff
            prior_boxes = self.prior_groups[i][j]
        ret = torch.zeros(self.gts.size(0))
        for start, end in self.intervals:
            truths = self.gts[start:end].cuda()
            overlaps = jaccard(truths, prior_boxes)
            best_prior_overlap, _ = overlaps.max(1, keepdim=False)
            ret[start:end] = best_prior_overlap
        return ret

    def __init__(self, dataset: (VOCDetection, COCODetection), prior_types_after_trim, cfg,
                 num_types_thresh, num_boxes_thresh, gts_cache=None,
                 num_samples=9999999, interval=4096, pool_size=40, weighted_layers=True):
        self.prior_types = prior_types_after_trim
        self.lweight = [t[:, -1].sigmoid().mean().item() for t in self.prior_types]
        print(self.lweight)
        self.pool_size = pool_size
        self.N = num_types_thresh
        self.M = num_boxes_thresh
        self.total_num_types = sum([o.size(0) for o in self.prior_types])
        self.feature_maps = cfg['feature_maps']
        self.__end_push = False
        # retrieve samples from the given dataset
        if gts_cache is None or not os.path.exists(gts_cache):
            sample_list = PriorsPool.__sample(dataset, num_samples)
            gts = []
            for sample in sample_list:
                objs, w, h = parse_rec(sample, with_size=True)
                mod = torch.tensor([1 / w, 1 / h, 1 / w, 1 / h])
                gts += [torch.stack([torch.tensor(obj['bbox'], dtype=torch.float) * mod for obj in objs])]
            gts = torch.cat(gts)
            self.gts = gts
            if gts_cache is not None:
                torch.save(self.gts, gts_cache)
        else:
            self.gts = torch.load(gts_cache)
        # generate prior boxes for each type. prior_groups[ k ][ l ] = prior boxes of l-th type in layer k
        self.prior_groups = PriorsPool.__gen_priors(self.prior_types, cfg)
        # intervals
        self.intervals = [i for i in range(0, self.gts.size(0), interval)] + [self.gts.size(0)]
        self.intervals = [(i, j) for i, j in zip(self.intervals[:-1], self.intervals[1:])]
        #
        self.selected_tokens = []
        self.num_selected_boxes = 0
        self.pool = {}
        # initial best IOUs, select the most important prior types of each layer
        self.best_ious = torch.zeros(self.gts.size(0))
        for i in range(len(self.prior_types)):
            token = i << 16
            self.best_ious = self.best_ious.max(self.__mk_iou_tensor(token))
            self.selected_tokens.append(token)
            self.num_selected_boxes += self.prior_groups[i][0].size(0)
        # initialize the pool
        for i in range(self.pool_size):
            self.__push()

    # 0 for functioning well
    # 1 for reaching the limit for the number of types or the limit for the number of prior boxes
    # 2 for no priors that can bring bonus regarding best IOUs with truths.
    def pop(self):
        if len(self.selected_tokens) >= self.N:
            return 1
        if self.num_selected_boxes >= self.M:
            return 1
        if len(self.pool) == 0:
            return 1
        # remove infeasible types
        flag = True
        while flag:
            abandon_lst = []
            for token in self.pool.keys():
                i, j = token >> 16, token & 0xffff
                if self.num_selected_boxes + self.prior_groups[i][j].size(0) > self.M:
                    abandon_lst.append(token)
            if len(abandon_lst) == 0:
                flag = False
            for token in abandon_lst:
                del self.pool[token]
                # self.pool.pop(token)
                self.__push()
        # find type with max bonus
        max_token = -1
        max_bonus = 1e-6
        for token, (ious, msk) in self.pool.items():
            # to reduce repeated calculations
            t, ot = ious[msk], self.best_ious[msk]
            mmsk = t > ot
            msk[msk] = mmsk
            bonus = (t[mmsk] - ot[mmsk]).sum().item() * self.lweight[token >> 16]
            if max_bonus < bonus:
                max_bonus = bonus
                max_token = token
        if max_token < 0:
            return 2
        # pop
        self.selected_tokens.append(max_token)
        ious, msk = self.pool[max_token]
        self.best_ious[msk] = ious[msk]
        i, j = max_token >> 16, max_token & 0xffff
        self.num_selected_boxes += self.prior_groups[i][j].size(0)
        print('select prior type %d in layer %d: %s' % (j, i, str(self.prior_types[i][j])))
        del self.pool[max_token]
        # self.pool.pop(max_token)
        self.__push()
        return 0

    def selected_prior_types(self):
        ret = [[] for _ in range(len(self.prior_types))]
        for token in self.selected_tokens:
            i, j = token >> 16, token & 0xffff
            ret[i].append(self.prior_types[i][j][:2])
        ret = [torch.stack(layer) for layer in ret]
        return ret

    def __push(self):
        if self.__end_push:
            return None
        if len(self.pool) >= self.pool_size:
            self.__end_push = True
            return None
        if len(self.pool) + len(self.selected_tokens) >= self.total_num_types:
            self.__end_push = True
            return None
        max_alpha = -99999999.
        # token value is supposed to be less than 2^31 - 1, while number of layers should be less than 2^15 - 1
        max_token = -1
        # find max
        for i, layer in enumerate(self.prior_types):
            for j, ptype in enumerate(layer):
                token = (i << 16) | j
                if token in self.selected_tokens or token in self.pool.keys():
                    continue
                if self.num_selected_boxes + self.prior_groups[i][j].size(0) > self.M:
                    continue
                if max_alpha < ptype[-1]:
                    max_alpha = ptype[-1]
                    max_token = token
        # No priors left
        if max_token < 0:
            self.__end_push = True
            return None
        # build record
        self.pool[max_token] = (self.__mk_iou_tensor(max_token), torch.ones(self.gts.size(0), dtype=torch_bool))
        return max_token
