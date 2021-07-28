import torch

from utils.anchor_generator_utils import mk_iou_tensor
from utils.box_utils import encode


class AnchorsAnalyzer:
    def __init__(self, anchors: torch.FloatTensor, gts: torch.FloatTensor):
        self._anchs = anchors
        self._gts = gts
        self._best_ious = None

    def get_recall(self, thresh=0.5):
        if self._best_ious is None:
            self._best_ious = mk_iou_tensor(self._anchs, self._gts)
        recall_t = self._best_ious > thresh
        recall = recall_t.sum().item() / recall_t.numel()
        return recall

    def get_power_mean(self, gamma):
        if self._best_ious is None:
            self._best_ious = mk_iou_tensor(self._anchs, self._gts)
        ret = (self._best_ious ** gamma).sum().item() / self._best_ious.numel()
        ret = ret ** (1 / gamma)
        return ret

    # power mean
    def get_specialty(self, gamma=3):
        return self.get_power_mean(gamma)

    def get_mean_best_ious(self):
        if self._best_ious is None:
            self._best_ious = mk_iou_tensor(self._anchs, self._gts)
        return self._best_ious.mean().item()

    def get_mean_best_gt_iou(self):
        best_gt_ious = mk_iou_tensor(self._gts, self._anchs)
        return best_gt_ious.mean().item()

    def get_approx_loss(self):
        self._best_ious, idx = mk_iou_tensor(self._anchs, self._gts, ret_idx=True)
        encoded_dis = encode(self._gts, self._anchs[idx], (0.1, 0.2))
        encoded_dis = torch.abs(encoded_dis)
        l1_tensor = torch.where(encoded_dis < 1., 0.5 * encoded_dis ** 2, encoded_dis - 0.5)
        l1 = l1_tensor.sum(dim=1)
        return l1.mean().item()

    def get_mean_log_ious(self):
        if self._best_ious is None:
            self._best_ious = mk_iou_tensor(self._anchs, self._gts)
        ret = self._best_ious.clamp_min(0.001).log().mean().item()
        return ret

    def get_geometric_mean_iou(self):
        if self._best_ious is None:
            self._best_ious: torch.Tensor = mk_iou_tensor(self._anchs, self._gts)
        ret = self._best_ious.clamp_min(0.001).log().mean()
        ret = torch.exp(ret).item()
        return ret

    def get_num_anchors(self):
        return self._anchs.size(0)
