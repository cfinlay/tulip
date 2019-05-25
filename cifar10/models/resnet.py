import torch as th
from torch import nn
from torch.nn.modules.utils import _pair

from . import blocks
from .blocks import Conv, Linear

class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)

class Avg2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        sh = x.shape
        x = x.contiguous().view(sh[0], sh[1], -1)
        return x.mean(-1)

class ResNet(nn.Module):

    def __init__(self, layers, block='BasicBlock', in_channels=3,
                 classes=10, kernel_size=(3,3),
                 conv0_kwargs = {'kernel_size':(3,3), 'stride':1},
                 conv0_pool=None, downsample_pool=nn.AvgPool2d,
                 base_channels=16,  nonlinear='relu'):
        super().__init__()
        kernel_size = _pair(kernel_size)

        def make_layer(n, block, in_channels, out_channels, stride):
            sublayers = []
            if not in_channels==out_channels:
                sublayers.append(Conv(in_channels, out_channels,
                    kernel_size=(1,1), nonlinear=nonlinear))

            if stride>1:
                sublayers.append(downsample_pool(stride))

            for k in range(n):
                sublayers.append(block(out_channels, kernel_size=kernel_size,
                    nonlinear=nonlinear))

            return nn.Sequential(*sublayers)


        block = getattr(blocks, block)

        self.layer0 = Conv(in_channels, base_channels, **conv0_kwargs, nonlinear=nonlinear)
        if conv0_pool:
            self.maxpool = conv0_pool
        else:
            self.maxpool = False


        _layers = []
        for i, n in enumerate(layers):

            if i==0:
                _layers.append(make_layer(n, block, base_channels, base_channels, 1))
            else:
                _layers.append(make_layer(n, block, base_channels*(2**(i-1)),
                    base_channels*(2**i), 2))

        self.layers = nn.Sequential(*_layers)

        self.pool = Avg2d()
        self.view = View((2**i)*base_channels)

        self.fc = Linear((2**i)*base_channels, classes, nonlinear=nonlinear)


    @property
    def num_parameters(self):
        return sum([w.numel() for w in self.parameters()])


    def forward(self, x):
        x = self.layer0(x)
        if self.maxpool:
            x = self.maxpool(x)
        x = self.layers(x)
        x = self.pool(x)
        x = self.view(x)
        x = self.fc(x)

        return x


def ResNeXt34_2x32(**kwargs):
    m = ResNet([5,5,5],block='BranchBlock',
            base_channels=32, **kwargs)
    return m

def SmoothResNeXt34_2x32(**kwargs):
    m = ResNet([5,5,5],block='BranchBlock',
            base_channels=32, nonlinear='c2relu',
            **kwargs)
    return m
