from layers.box_utils import jaccard, point_form
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePriorBoxesLoss(nn.Module):
    def __init__(self, beta=1., k=2.5, iou_thresh=0.4):
        super(AdaptivePriorBoxesLoss, self).__init__()
        self.beta = beta
        self.k = k
        self.thresh = iou_thresh

    def forward(self, locs: torch.Tensor, params: torch.Tensor, truths):
        sigmoid_alphas = params[:, -1].sigmoid()          # size [num_priors]
        priors = torch.cat([locs, params[:, :2]], dim=1)  # size [num_priors, 4]
        overlaps = jaccard(
            truths,
            point_form(priors)
        )  # size [num_truths, num_priors]
        # [num_priors] best ground truth for each prior
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=False)
        # [1,num_objects] best prior for each ground truth
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=False)
        best_truth_overlap[best_prior_idx] = best_prior_overlap
        # filter
        x_filter = torch.zeros(best_truth_overlap.size())
        x_filter[best_truth_overlap > self.thresh] = 1.
        x_filter[best_prior_idx] = self.k
        return (-(sigmoid_alphas * x_filter * best_truth_overlap.log()).sum()
                + self.beta * sigmoid_alphas.sum()) / x_filter.sum()
