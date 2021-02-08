from torch import nn


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6, padding=1, dilation=1, output2=False):
        super(InvertedResidualBlock, self).__init__()
        assert stride in [1, 2]
        self._output2 = output2
        e_in = in_channels * expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, e_in, kernel_size=1, bias=False),
            nn.BatchNorm2d(e_in),
            nn.ReLU6(inplace=True)
        )
        self.dwise_conv = nn.Sequential(
            nn.Conv2d(e_in, e_in, kernel_size=3, stride=stride, padding=padding,
                      dilation=dilation, groups=e_in, bias=False),
            nn.BatchNorm2d(e_in),
            nn.ReLU6(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(e_in, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expand(x)
        out2 = out
        out = self.dwise_conv(out)
        out = self.project(out)
        if out.shape == x.shape:
            out = out + x
        if self._output2:
            return out, out2
        return out


class FollowedDownSampleBlock(nn.Module):
    def __init__(self, in_channels, medium_channels, out_channels, padding=1):
        super(FollowedDownSampleBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, medium_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(medium_channels),
            nn.ReLU6(inplace=True)
        )
        self.dw_conv = nn.Sequential(
            nn.Conv2d(medium_channels, medium_channels, kernel_size=3, stride=2,
                      padding=padding, groups=medium_channels, bias=False),
            nn.BatchNorm2d(medium_channels),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(medium_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.dw_conv(out)
        out = self.conv2(out)
        return out


class MConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: [int, tuple] = 3, stride=1, padding=0, dilation=1,
                 bias=False, padding_mode='zeros'):
        super(MConv, self).__init__()
        self.conv_dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                      dilation, in_channels, bias, padding_mode),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_1x1(self.conv_dw(x))


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: [int, tuple], stride=1, padding=0, dilation=1,
                 groups=1, bias=False, padding_mode='zeros'):
        super(ConvBnReLU, self).__init__()
        self.my_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation, groups, bias, padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.my_conv(x)
