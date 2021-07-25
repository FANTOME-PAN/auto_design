from utils.anchor_generator_utils import mk_iou_tensor
from data.voc0712 import VOCDetection
import torch


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

    # power mean
    def get_specialty(self, gamma=6, thresh=0.5):
        if self._best_ious is None:
            self._best_ious = mk_iou_tensor(self._anchs, self._gts)
        valid_ious = self._best_ious[self._best_ious > thresh]
        sp = (valid_ious ** gamma).sum().item() / valid_ious.numel()
        sp = sp ** (1 / gamma)
        return sp

    def get_mean_best_ious(self):
        if self._best_ious is None:
            self._best_ious = mk_iou_tensor(self._anchs, self._gts)
        return self._best_ious.mean().item()


