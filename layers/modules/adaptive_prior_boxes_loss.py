from layers.box_utils import jaccard, point_form
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePriorBoxesLoss(nn.Module):
    def __init__(self, beta):
        super(AdaptivePriorBoxesLoss, self).__init__()
        self.beta = beta

    def forward(self, locs: torch.Tensor, params: torch.Tensor, truths):
        sigmoid_alphas = F.sigmoid(params[:, -1])  # size [num_priors]
        priors = torch.cat([locs, params], dim=1)  # size [num_priors, 4]
        overlaps = jaccard(
            truths,
            point_form(priors)
        )  # size [num_truths, num_priors]
        # [num_priors] best ground truth for each prior
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=False)
        return self.beta * sigmoid_alphas.sum().log() \
            - (sigmoid_alphas * best_truth_overlap).sum().log()
