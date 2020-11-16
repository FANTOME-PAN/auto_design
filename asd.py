import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.functions.prior_box import ASDPriorBox
from layers import *
from data.config import voc_sd
import os


class AutoTailoredSmallDetector(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes, cfg):
        super(AutoTailoredSmallDetector, self).__init__()
        self.phase = phase
        assert num_classes == 2
        self.num_classes = num_classes
        self.cfg = cfg
        self.priorbox = ASDPriorBox(self.cfg)
        self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.conf_net_input = None

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources, loc, conf = [], [], []
        sur_x = None
        for i, v in enumerate(self.base):
            x = v(x)
            if i in [5, 8, 11]:
                sur_x = x
            if i in [6, 9, 12]:
                x += sur_x
            if i in self.cfg['base_output_layers']:
                sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k in self.config['extras_output_layers']:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            tmp = self.softmax(conf.view(conf.size(0), -1, self.num_classes))
            self.conf_net_input = tmp
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                tmp,  # conf preds
                self.priors.type(x.dtype)  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def base_net_forward(self, x):
        for v in self.base:
            x = v(x)
        return x

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_asd(phase, size=300, ratio_3x3=0.5, num_classes=2, cfg=None):
    classifier_chnls = {38: 256, 19: 256, 10: 256, 5: 256, 3: 256, 1: 256}
    from math import ceil as mc, floor as mf
    from layers.squeeze_net import FireModule
    p, q = 1 - ratio_3x3, ratio_3x3
    base_net = [
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        FireModule(32, 16, mc(128 * p), mf(128 * q)),       # 150 x 150
        FireModule(128, 32, mc(256 * p), mf(256 * q)),
        nn.MaxPool2d(2),
        FireModule(256, 32, mc(256 * p), mf(256 * q)),      # 75 x 75
        FireModule(256, 48, mc(384 * p), mf(384 * q)),
        nn.MaxPool2d(2, 2, 1),
        FireModule(384, 48, mc(384 * p), mf(384 * q)),      # 38 x 38
        FireModule(384, 64, mc(512 * p), mf(512 * q)),
        nn.MaxPool2d(2),
        FireModule(512, 64, mc(512 * p), mf(512 * q))       # 19 x 19
    ]
    extras = [
        nn.Conv2d(512, 256, 1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, 2, 1), nn.ReLU(inplace=True),    # 3 Conv8_2  10x10
        nn.Conv2d(256, 128, 1), nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),    # 7 Conv9_2  5x5
        nn.Conv2d(256, 128, 1), nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, 3), nn.ReLU(inplace=True),          # 11 Conv10_2 3x3
        nn.Conv2d(256, 128, 1), nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, 3), nn.ReLU(inplace=True),          # 15 Conv11_2 1x1
    ]
    head = [
        [nn.Conv2d(classifier_chnls[feature], cfg['num_prior_boxes'][i] * 4, kernel_size=3, padding=1)
         for i, feature in enumerate(cfg['feature_maps'])],
        [nn.Conv2d(classifier_chnls[feature], cfg['num_prior_boxes'][i] * num_classes, kernel_size=3, padding=1)
         for i, feature in enumerate(cfg['feature_maps'])]
    ]
    return AutoTailoredSmallDetector(phase, size, base_net, extras, head, num_classes, cfg)




