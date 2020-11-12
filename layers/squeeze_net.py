import torch
from torch import nn
import torch.nn.functional as F


class FireModule(nn.Module):
    def __init__(self, in_channels, s_1x1, e_1x1, e_3x3):
        super(FireModule, self).__init__()
        if s_1x1 >= e_1x1 + e_3x3:
            raise Warning('too many squeeze filters')
        self.s1 = nn.Conv2d(in_channels, s_1x1, kernel_size=1)
        self.e1 = nn.Conv2d(s_1x1, e_1x1, kernel_size=1)
        self.e3 = nn.Conv2d(s_1x1, e_3x3, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.s1(x), inplace=True)
        out = torch.cat([self.e1(out), self.e3(out)], 1)
        return F.relu(out, inplace=True)


