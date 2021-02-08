import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from data import helmet_lite
from data.config import voc_lite, voc
import os
from layers.mobilenets import InvertedResidualBlock, FollowedDownSampleBlock, MConv, ConvBnReLU


class SmallSSD(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes, cfg=None):
        super(SmallSSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg if cfg is not None else helmet_lite
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.base = self.vgg

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

    def basenet_forward(self, x):
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


class TestSmallSSD(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes, cfg=helmet_lite):
        super(TestSmallSSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.conf_net_input = None

        self.softmax = nn.Softmax(dim=-1)

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
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            self.priors
        )
        return output

    def basenet_forward(self, x):
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


# class MobileSSD(nn.Module):
#     def __init__(self, phase, size, base, extras, head, num_classes, cfg=None):
#         super(MobileSSD, self).__init__()
#         self.phase = phase
#         self.num_classes = num_classes
#         self.cfg = cfg
#         self.priorbox = PriorBox(self.cfg if cfg is not None else voc_lite)
#         self.priors = self.priorbox.forward()
#         self.size = size
#
#         # SSD network
#         self.base = nn.ModuleList(base)
#         self.extras = nn.ModuleList(extras)
#
#         self.loc = nn.ModuleList(head[0])
#         self.conf = nn.ModuleList(head[1])
#
#         self.conf_net_input = None
#
#         self.softmax = nn.Softmax(dim=-1)
#         self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
#
#     def forward(self, x):
#         sources, loc, conf = [], [], []
#
#         for v in self.base:
#             x = v(x)
#         sources.append(x)
#
#         # apply extra layers and cache source layer outputs
#         for k, v in enumerate(self.extras):
#             x = F.relu(v(x), inplace=True)
#             if k % 2 == 1:
#                 sources.append(x)
#
#         # apply multibox head to source layers
#         for (x, l, c) in zip(sources, self.loc, self.conf):
#             loc.append(l(x).permute(0, 2, 3, 1).contiguous())
#             conf.append(c(x).permute(0, 2, 3, 1).contiguous())
#
#         loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
#         conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
#         if self.phase == "test":
#             tmp = self.softmax(conf.view(conf.size(0), -1, self.num_classes))
#             self.conf_net_input = tmp
#             output = self.detect(
#                 loc.view(loc.size(0), -1, 4),  # loc preds
#                 tmp,  # conf preds
#                 self.priors.type(x.dtype)  # default boxes
#             )
#         else:
#             output = (
#                 loc.view(loc.size(0), -1, 4),
#                 conf.view(conf.size(0), -1, self.num_classes),
#                 self.priors
#             )
#         return output
#
#     def basenet_forward(self, x):
#         for v in self.base:
#             x = v(x)
#         return x
#
#     def load_weights(self, base_file):
#         other, ext = os.path.splitext(base_file)
#         if ext == '.pkl' or '.pth':
#             print('Loading weights into state dict...')
#             self.load_state_dict(torch.load(base_file,
#                                  map_location=lambda storage, loc: storage))
#             print('Finished!')
#         else:
#             print('Sorry only .pth and .pkl files supported.')
class MobileNetSSD(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes, cfg=None):
        super(MobileNetSSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(cfg if cfg is not None else voc)
        self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.__base_output = [i - k - 1 for k, i in enumerate([j for j, o in enumerate(base) if o == 'O'])]
        self.base = nn.ModuleList([o for o in base if o != 'O'])
        self.__extras_output = [i - k - 1 for k, i in enumerate([j for j, o in enumerate(extras) if o == 'O'])]
        self.extras = nn.ModuleList([o for o in extras if o != 'O'])

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.conf_net_input = None

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources, loc, conf = [], [], []

        # apply base net layers and cache source layer outputs
        for k, v in enumerate(self.base):
            x = v(x)
            if k in self.__base_output:
                sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k in self.__extras_output:
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

    def basenet_forward(self, x):
        out = x
        for v in self.base:
            out = v(out)
        return out

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


class MobileNetV2SSD(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes, cfg=None):
        super(MobileNetV2SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(cfg if cfg is not None else voc)
        self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.__base_output_dict = dict([(i - k - 1, d) for k, (i, d) in
                                        enumerate([(j, o) for j, o in enumerate(base) if o in ['O', 'O2']])])
        self.base = nn.ModuleList([o for o in base if o not in ['O', 'O2']])
        self.__extras_output_dict = dict([(i - k - 1, d) for k, (i, d) in
                                          enumerate([(j, o) for j, o in enumerate(extras) if o in ['O', 'O2']])])
        self.extras = nn.ModuleList([o for o in extras if o not in ['O', 'O2']])

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.conf_net_input = None

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources, loc, conf = [], [], []

        # apply base net layers and cache source layer outputs
        for k, v in enumerate(self.base):
            x = v(x)
            if k in self.__base_output_dict.keys():
                if self.__base_output_dict[k] == 'O':
                    sources.append(x)
                else:
                    # O2, i.e. output2 is set to True, output of the block would be (out, out2).
                    # out2 is the medium output of expansion part.
                    sources.append(x[1])
                    x = x[0]

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k in self.__extras_output_dict.keys():
                if self.__extras_output_dict[k] == 'O':
                    sources.append(x)
                else:
                    # THIS SNIPPET WON'T BE EXECUTED, because extra layers do not contain any InvertedResidualBlock.
                    # O2, i.e. output2 is set to True, output of the block would be (out, out2).
                    # out2 is the medium output of expansion part.
                    sources.append(x[1])
                    x = x[0]
                    raise RuntimeError('Unexpected Position')

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

    def basenet_forward(self, x):
        out = x
        for v in self.base:
            out = v(out)
        return out

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def get_small_ssd_params(num_classes):
    vgg = [
        nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(inplace=True),

        nn.MaxPool2d(3, 1, 1),  # is it necessary?
        nn.Conv2d(512, 512, 3, padding=6, dilation=6), nn.ReLU(inplace=True),
        nn.Conv2d(512, 1024, 1, 1, 0), nn.ReLU(inplace=True)
    ]
    extras = [
        nn.Conv2d(1024, 256, 1),
        nn.Conv2d(256, 512, 3, 2, 1),  # Conv8_2
        nn.Conv2d(512, 128, 1),
        nn.Conv2d(128, 256, 3, 2, 1),  # Conv9_2
        nn.Conv2d(256, 128, 1),
        nn.Conv2d(128, 256, 3),  # Conv10_2
        nn.Conv2d(256, 128, 1),
        nn.Conv2d(128, 256, 3),  # Conv11_2
    ]
    head = [
        # loc layers
        [
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        ],
        # conf layers
        [
            nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)
        ]
    ]
    return vgg, extras, head


def build_small_ssd(phase, size=300, num_classes=5, cfg=None):
    vgg, extras, head = get_small_ssd_params(num_classes)
    return SmallSSD(phase, size, vgg, extras, head, num_classes, cfg)


# def build_mobile_ssd(phase, size=300, num_classes=5, cfg=None):
#     base = [
#         nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
#         nn.MaxPool2d(2),
#         V2Block(64, 32, expansion=1),
#         V2Block(32, 48, stride=2),
#         V2Block(48, 64, stride=2),
#         V2Block(64, 128, stride=2),
#
#         nn.MaxPool2d(3, 1, 1),  # why is it necessary?
#         V2Block(128, 1024, padding=6, dilation=6),
#         nn.Conv2d(1024, 1024, 1, 1, 0), nn.ReLU(inplace=True)
#     ]
#     extras = [
#         nn.Conv2d(1024, 256, 1),
#         nn.Conv2d(256, 512, 3, 2, 1),  # Conv8_2
#         nn.Conv2d(512, 128, 1),
#         nn.Conv2d(128, 256, 3, 2, 1),  # Conv9_2
#         nn.Conv2d(256, 128, 1),
#         nn.Conv2d(128, 256, 3),        # Conv10_2
#         nn.Conv2d(256, 128, 1),
#         nn.Conv2d(128, 256, 3),        # Conv11_2
#     ]
#     head = [
#         # loc layers
#         [
#             nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
#             nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
#             nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
#             nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
#             nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
#         ],
#         # conf layers
#         [
#             nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)
#         ]
#     ]
#     return MobileSSD(phase, size, base, extras, head, num_classes, cfg)


# def build_mobile_ssd_v2(phase, size=300, num_classes=5, cfg=None):
#     base = [
#         nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(inplace=True),
#         V2Block(32, 16, expansion=1),
#
#         V2Block(16, 24, stride=2),
#         V2Block(24, 24, stride=1),
#
#         V2Block(24, 32, stride=2),
#         V2Block(32, 32, stride=1),
#         V2Block(32, 32, stride=1),
#
#         V2Block(32, 64, stride=2),
#         V2Block(64, 64, stride=1),
#         V2Block(64, 64, stride=1),
#         V2Block(64, 64, stride=1),
#
#         V2Block(64, 96, stride=1),
#         V2Block(96, 96, stride=1),
#         V2Block(96, 96, stride=1),
#
#         V2Block(96, 128, stride=1),
#         V2Block(128, 128, stride=1),
#         V2Block(128, 128, stride=1),
#
#         V2Block(128, 256, stride=1),
#         V2Block(256, 1024, stride=1, dilation=2, padding=2),
#         nn.Conv2d(1024, 1024, 1, 1, 0), nn.ReLU(inplace=True)
#     ]
#     extras = [
#         nn.Conv2d(1024, 256, 1),
#         nn.Conv2d(256, 512, 3, 2, 1),  # Conv8_2
#         nn.Conv2d(512, 128, 1),
#         nn.Conv2d(128, 256, 3, 2, 1),  # Conv9_2
#         nn.Conv2d(256, 128, 1),
#         nn.Conv2d(128, 256, 3),        # Conv10_2
#         nn.Conv2d(256, 128, 1),
#         nn.Conv2d(128, 256, 3),        # Conv11_2
#     ]
#     head = [
#         # loc layers
#         [
#             nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
#             nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
#             nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
#             nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
#             nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
#         ],
#         # conf layers
#         [
#             nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)
#         ]
#     ]
#     return MobileSSD(phase, size, base, extras, head, num_classes, cfg)

def build_mobilenet_ssd(phase, size=300, num_classes=5, cfg=None):
    base = [
        nn.Conv2d(3, 32, 3, 1, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        MConv(32, 64, stride=1, padding=1),

        MConv(64, 128, stride=2, padding=1),
        MConv(128, 128, stride=1, padding=1),

        MConv(128, 256, stride=2, padding=1),
        MConv(256, 256, stride=1, padding=1),

        MConv(256, 512, stride=2, padding=1),
        MConv(512, 512, stride=1, padding=1),
        MConv(512, 512, stride=1, padding=1),
        MConv(512, 512, stride=1, padding=1),
        MConv(512, 512, stride=1, padding=1),
        MConv(512, 512, stride=1, padding=1), 'O',

        MConv(512, 1024, stride=2, padding=1),
        MConv(1024, 1024, stride=1, padding=1), 'O'
    ]
    extras = [
        ConvBnReLU(1024, 256, 1),
        ConvBnReLU(256, 512, 3, 2, 1), 'O',  # Conv8_2
        ConvBnReLU(512, 128, 1),
        ConvBnReLU(128, 256, 3, 2, 1), 'O',  # Conv9_2
        ConvBnReLU(256, 128, 1),
        ConvBnReLU(128, 256, 3), 'O',  # Conv10_2
        ConvBnReLU(256, 128, 1),
        ConvBnReLU(128, 256, 3), 'O',  # Conv11_2
    ]
    head = [
        # loc layers
        [
            nn.Conv2d(512, 4 * 4, kernel_size=1),
            nn.Conv2d(1024, 6 * 4, kernel_size=1),
            nn.Conv2d(512, 6 * 4, kernel_size=1),
            nn.Conv2d(256, 6 * 4, kernel_size=1),
            nn.Conv2d(256, 4 * 4, kernel_size=1),
            nn.Conv2d(256, 4 * 4, kernel_size=1)
        ],
        # conf layers
        [
            nn.Conv2d(512, 4 * num_classes, kernel_size=1),
            nn.Conv2d(1024, 6 * num_classes, kernel_size=1),
            nn.Conv2d(512, 6 * num_classes, kernel_size=1),
            nn.Conv2d(256, 6 * num_classes, kernel_size=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=1)
        ]
    ]
    return MobileNetSSD(phase, size, base, extras, head, num_classes, cfg)


def build_mobilenet_v2_ssd(phase, size=300, num_classes=5, cfg=None):
    base = [
        # stage 1
        nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        ),
        InvertedResidualBlock(32, 16, expansion=1),
        # stage 2
        InvertedResidualBlock(16, 24, stride=2),
        InvertedResidualBlock(24, 24),
        # stage 3
        InvertedResidualBlock(24, 32, stride=2),
        InvertedResidualBlock(32, 32),
        InvertedResidualBlock(32, 32),
        # stage 4
        InvertedResidualBlock(32, 64, stride=2),
        InvertedResidualBlock(64, 64),
        InvertedResidualBlock(64, 64),
        InvertedResidualBlock(64, 64),
        InvertedResidualBlock(64, 96),
        InvertedResidualBlock(96, 96),
        InvertedResidualBlock(96, 96),
        # stage 5
        InvertedResidualBlock(96, 160, stride=2, output2=True), 'O2',  # 19x19
        InvertedResidualBlock(160, 160),
        InvertedResidualBlock(160, 160),
        InvertedResidualBlock(160, 320),
        nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        ), 'O',  # 10x10
    ]
    extras = [
        FollowedDownSampleBlock(1280, 256, 512, padding=1), 'O',  # 5x5
        FollowedDownSampleBlock(512, 128, 256, padding=1), 'O',  # 3x3
        FollowedDownSampleBlock(256, 128, 256, padding=0), 'O',  # 1x1
        FollowedDownSampleBlock(256, 64, 128, padding=1), 'O',  # 1x1
    ]

    def predict_blk(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    num_boxes = cfg['num_prior_boxes']
    head = [
        # loc layers
        [
            predict_blk(576, num_boxes[0] * 4),
            predict_blk(1280, num_boxes[1] * 4),
            predict_blk(512, num_boxes[2] * 4),
            predict_blk(256, num_boxes[3] * 4),
            predict_blk(256, num_boxes[4] * 4),
            predict_blk(128, num_boxes[5] * 4)
        ],
        # conf layers
        [
            predict_blk(576, num_boxes[0] * num_classes),
            predict_blk(1280, num_boxes[1] * num_classes),
            predict_blk(512, num_boxes[2] * num_classes),
            predict_blk(256, num_boxes[3] * num_classes),
            predict_blk(256, num_boxes[4] * num_classes),
            predict_blk(128, num_boxes[5] * num_classes),
        ]
    ]
    return MobileNetV2SSD(phase, size, base, extras, head, num_classes, cfg)


if __name__ == '__main__':
    # m = TestSmallSSD('train', 300, *get_small_ssd_params(21), 21, voc_lite)
    m = nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )
    # _test_in = torch.randn(1, 3, 300, 300)
    # from thop import profile
    # print(profile(m, (_test_in,)))
    # from pthflops import count_ops
    # print(count_ops(m, _test_in))
