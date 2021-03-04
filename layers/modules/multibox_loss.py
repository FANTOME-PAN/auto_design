# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, best_prior_weight=5.,
                 use_gpu=True, knowledge_distill=False, use_half=False):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self._enable_distill = knowledge_distill
        self.use_half = use_half
        self.bpw = best_prior_weight

    def forward(self, predictions, targets, big_ssd_preds=None, distill_mask=None):
        loc_data, conf_data, priors = predictions
        if self._enable_distill:
            assert big_ssd_preds is not None
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        # loc_t = torch.zeros(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        # conf_t = torch.
        best_priors_msk = []
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            if self.use_half:
                truths = truths.half()
            defaults = priors.data
            pmsk = match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
            best_priors_msk.append(pmsk)
        # best_priors_msk = torch.stack(best_priors_msk)
        best_priors_msk = torch.zeros(num, num_priors, dtype=torch.uint8)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t.requires_grad = False
        conf_t.requires_grad = False

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        assert (pos & best_priors_msk == best_priors_msk).min().item() == 1
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos].view(-1, 4)
        loc_t = loc_t[pos].view(-1, 4)
        pos_idx_l = pos
        msk = best_priors_msk[pos]
        t1, t2, t3, t4 = loc_p[~msk], loc_t[~msk], loc_p[msk], loc_t[msk]
        tt1 = torch.isnan(t1).sum()
        tt2 = torch.isnan(t2).sum()
        tt3 = torch.isinf(t1).sum()
        tt4 = torch.isinf(t2).sum()
        loss_l = F.smooth_l1_loss(loc_p[~msk], loc_t[~msk], reduction='sum')
        # loss_l += self.bpw * F.smooth_l1_loss(loc_p[msk], loc_t[msk], reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(pos.size(0), pos.size(1))
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        # pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        chosen_idx = pos | neg
        # chosen_idx = pos_idx | neg_idx
        conf_p = conf_data[chosen_idx].view(-1, self.num_classes)
        targets_weighted = conf_t[pos | neg]
        msk = best_priors_msk[pos | neg]
        t1, t2, t3, t4 = conf_p[~msk], targets_weighted[~msk], conf_p[msk], targets_weighted[msk]
        loss_c = F.cross_entropy(conf_p[~msk], targets_weighted[~msk], reduction='sum')
        # loss_c += self.bpw * F.cross_entropy(conf_p[msk], targets_weighted[msk], reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = pos.sum() + best_priors_msk.sum() * (self.bpw - 1.)
        loss_l /= N
        loss_c /= N
        if self._enable_distill:
            big_loc_data, big_conf_data, _ = big_ssd_preds
            inv_temperature = 1 / 1.
            # 大网络和小网络的 prior boxes 数量不同。vgg-lite小网络没有对 38 * 38 的识别。
            if distill_mask is not None:
                big_loc_data = big_loc_data[:, distill_mask]
                big_conf_data = big_conf_data[:, distill_mask]
            big_conf_p = big_conf_data[chosen_idx].view(-1, self.num_classes)
            y_softmax = F.log_softmax(conf_p * inv_temperature, dim=1)
            y_big_softmax = F.softmax(big_conf_p, dim=1)

            big_loc_p = big_loc_data[pos_idx_l].view(-1, 4)
            # same as loss_c and loss_l
            loss_c_distill = -(y_big_softmax * y_softmax).sum(dim=1).sum()
            loss_l_distill = F.smooth_l1_loss(loc_p, big_loc_p, reduction='sum')
            loss_c_distill /= N
            loss_l_distill /= N
            return loss_l, loss_c, loss_c_distill, loss_l_distill
        return loss_l, loss_c


