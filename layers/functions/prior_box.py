from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class AdaptivePriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg, times_var):
        super(AdaptivePriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        self.intervals = [0] + [k * k * (2 + 2 * len(o)) * times_var for k, o in zip(self.feature_maps, self.aspect_ratios)]
        for k in range(len(self.intervals) - 1):
            self.intervals[k + 1] += self.intervals[k]
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    # params is the list of tensors, the length of the list must be equal to the number of feature maps.
    # each tensor in params is [N, 3] in size, N is the number of priors, 3 refers to height, width and alpha value.
    def forward(self, params, requires_grad=True):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                for p in params[k]:
                    tmp = torch.zeros(5)
                    tmp[0], tmp[1] = cx, cy
                    tmp[2:] = p
                    mean += [tmp]
        # back to torch land
        output = torch.stack(mean)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

    # do not contains loc.
    def fast_forward(self, params, alphas, output=None):
        if output is None:
            output = torch.zeros(self.intervals[-1], 3)
        for k, num in enumerate(self.feature_maps):
            p = params[k].unsqueeze(0).expand(num * num, params[k].size(0), 3)
            tp_out = output[self.intervals[k]: self.intervals[k + 1], :].view_as(p)
            tp_out[:] = p
        output[:, -1] = alphas
        return output


class ASDPriorBox(PriorBox):
    def __init__(self, cfg):
        super(ASDPriorBox, self).__init__(cfg)

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    # cx, cy, width, height
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
        # back to torch land
        output = torch.tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
