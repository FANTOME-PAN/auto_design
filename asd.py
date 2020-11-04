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
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.conf_net_input = None

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources, loc, conf = [], [], []

        for v in self.vgg:
            x = v(x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
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

    def vgg_forward(self, x):
        for v in self.vgg:
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


def build_asd(phase, size=300, num_classes=2, cfg=voc_sd):
    classifier_chnls = {38: 256, 19: 256, 10: 256, 5: 256, 3: 256, 1: 256}
    base = [
        nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),    # 150
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),    # 75
        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),    # 38
        nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),    # 38*38 output
        nn.MaxPool2d(2),    # 19
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 1, 1, 0), nn.ReLU(inplace=True)     # 19*19 output
    ]
    extras = [
        nn.Conv2d(256, 256, 1),
        nn.Conv2d(256, 256, 3, 2, 1),  # 1 Conv8_2
        nn.Conv2d(256, 128, 1),
        nn.Conv2d(128, 256, 3, 2, 1),  # 3 Conv9_2
        nn.Conv2d(256, 128, 1),
        nn.Conv2d(128, 256, 3),        # 5 Conv10_2
        nn.Conv2d(256, 128, 1),
        nn.Conv2d(128, 256, 3),        # 7 Conv11_2
    ]
    head = [
        [nn.Conv2d(classifier_chnls[feature], cfg['num_prior_boxes'][i] * 4, kernel_size=3, padding=1)
         for i, feature in enumerate(cfg['feature_maps'])],
        [nn.Conv2d(classifier_chnls[feature], cfg['num_prior_boxes'][i] * num_classes, kernel_size=3, padding=1)
         for i, feature in enumerate(cfg['feature_maps'])]
    ]
    return AutoTailoredSmallDetector(phase, size, base, extras, head, num_classes, cfg)




