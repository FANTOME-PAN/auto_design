from layers.box_utils import encode, jaccard, point_form
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePriorBoxesLoss(nn.Module):
    # Restraint: iou threshold >= 0.5
    def __init__(self, beta=1., k=5, iou_thresh=0.5):
        super(AdaptivePriorBoxesLoss, self).__init__()
        self.beta = beta
        self.k = k
        self.thresh = iou_thresh

    def forward(self, locs: torch.Tensor, params: torch.Tensor, truths, variance=(0.1, 0.2)):
        sigmoid_alphas = params[:, -1].sigmoid()  # size [num_priors]
        priors = torch.cat([locs, params[:, :2]], dim=1)  # size [num_priors, 4]
        with torch.no_grad():
            overlaps = jaccard(
                truths,
                point_form(priors)
            )  # size [num_truths, num_priors]
        # [num_priors] best ground truth for each prior
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=False)
        # [1,num_objects] best prior for each ground truth
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=False)
        # replace original best truth indexes whose prior boxes are the best priors of given truths
        best_truth_overlap[best_prior_idx] = best_prior_overlap
        best_truth_idx[best_prior_idx] = torch.tensor(range(best_prior_idx.size(0)), dtype=torch.long)
        # create filter
        x_filter = torch.zeros(best_truth_overlap.size())
        x_filter[best_truth_overlap > self.thresh] = 1.
        x_filter[best_prior_idx] = self.k
        # filtering
        msk = x_filter > 1e-7
        x_filter = x_filter[msk]
        best_truth_overlap = best_truth_overlap[msk]
        # return loss value
        return ((sigmoid_alphas[msk] * x_filter * best_truth_overlap).sum()
                + self.beta * sigmoid_alphas.sum()) / x_filter.sum()


class AdaptivePBLossDebug(AdaptivePriorBoxesLoss):
    # Restraint: iou threshold >= 0.5
    def __init__(self, beta=1., k=5, iou_thresh=0.5):
        super(AdaptivePBLossDebug, self).__init__()
        self.beta = beta
        self.k = k
        self.thresh = iou_thresh
        print('Class AdaptivePBLossDebug instantiated:{\n'
              '\t\'beta\' = %.2f\n\t\'k\' = %.2f\n\t\'iou threshold\' = %.2f\n}'
              % (self.beta, self.k, self.thresh))

    def forward(self, locs: torch.Tensor, params: torch.Tensor, truths, variance=(0.1, 0.2)):
        sigmoid_alphas = params[:, -1].sigmoid()  # size [num_priors]
        priors = torch.cat([locs, params[:, :2]], dim=1)  # size [num_priors, 4]
        with torch.no_grad():
            overlaps = jaccard(
                truths,
                point_form(priors)
            )  # size [num_truths, num_priors]
        # [num_priors] best ground truth for each prior
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=False)
        # [1,num_objects] best prior for each ground truth
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=False)
        # replace original best truth indexes whose prior boxes are the best priors of given truths
        best_truth_overlap[best_prior_idx] = best_prior_overlap
        best_truth_idx[best_prior_idx] = torch.tensor(range(best_prior_idx.size(0)), dtype=torch.long)
        # create filter
        x_filter = torch.zeros(best_truth_overlap.size())
        x_filter[best_truth_overlap > self.thresh] = 1.
        x_filter[best_prior_idx] = self.k
        # filtering
        msk = x_filter > 1e-7
        x_filter = x_filter[msk]
        best_truth_overlap = best_truth_overlap[msk]
        # log info
        aaa = (best_truth_overlap < 1e-7).sum().item()
        print('%d best truths after filtering' % (x_filter > 1e-4).sum().item())
        print('%d best priors, of which %d priors fail to meet iou threshold'
              % (best_prior_idx.size(0), (best_prior_overlap <= self.thresh).sum().item()))
        ret = ((sigmoid_alphas[msk] * x_filter * best_truth_overlap).sum()
               + self.beta * sigmoid_alphas.sum()) / x_filter.sum()
        print("loss fn: (%.2f(1st term) + %.2f(2nd term)) / %.2f(3rd term) = %.2f"
              % ((sigmoid_alphas[msk] * x_filter * best_truth_overlap).sum().item(),
                 self.beta * sigmoid_alphas.sum().item(),
                 x_filter.sum().item(), ret.item()))
        # return loss value
        return ret


class L1EncodedLoss(AdaptivePriorBoxesLoss):
    def forward(self, locs: torch.Tensor, params: torch.Tensor, truths, variance=(0.1, 0.2)):
        sigmoid_alphas = params[:, -1].sigmoid()  # size [num_priors]
        priors = torch.cat([locs, params[:, :2]], dim=1)  # size [num_priors, 4]
        with torch.no_grad():
            overlaps = jaccard(
                truths,
                point_form(priors)
            )  # size [num_truths, num_priors]
        # [num_priors] best ground truth for each prior
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=False)
        # [1,num_objects] best prior for each ground truth
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=False)
        # replace original best truth indexes whose prior boxes are the best priors of given truths
        # best_truth_overlap[best_prior_idx] = best_prior_overlap
        best_truth_idx[best_prior_idx] = torch.tensor(range(best_prior_idx.size(0)), dtype=torch.long)
        # filter
        x_filter = torch.zeros(best_truth_overlap.size())
        x_filter[best_truth_overlap > self.thresh] = 1.
        x_filter[best_prior_idx] = self.k
        # encode, L1_loss
        encoded_dis = encode(truths[best_truth_idx], priors, variance)
        encoded_dis = torch.abs(encoded_dis)
        l1_tensor = torch.where(encoded_dis < 1., 0.5 * encoded_dis ** 2, encoded_dis - 0.5)
        l1_tensor = l1_tensor.sum(dim=1)
        # return loss value
        return ((sigmoid_alphas * x_filter * l1_tensor).sum() + self.beta * sigmoid_alphas.sum()) / x_filter.sum()
