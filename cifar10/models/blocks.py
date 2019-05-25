import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
import math

def c2relu(x):
    z = F.relu(x+0.5,inplace=True)
    y = F.relu(x,inplace=True)
    return th.max(-0.5*z.pow(4) + z.pow(3), 
                  y)

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim,  bn=True, nonlinear='relu', 
            bias=True, **kwargs):
        """A linear block.  The linear layer is followed by batch
        normalization and a ReLU (if active).

        Args:
            in_dim: number of input dimensions
            out_dim: number of output dimensions
            bn (bool, optional): turn on batch norm (default: False)
        """
        super().__init__()

        self.weight = nn.Parameter(th.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(th.randn(out_dim))
        else:
            self.register_parameter('bias', None)
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.nonlinear=nonlinear

        if bn:
            self.bn = nn.BatchNorm1d(out_dim, affine=False)
        else:
            self.bn = False


        self.reset_parameters()



    def reset_parameters(self):
        n = self.in_dim
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        y = F.linear(x, self.weight, None)

        if self.bn:
            y = self.bn(y)

        if self.bias is not None:
            b = self.bias.view(1,-1)
            y = y+b

        if self.nonlinear=='relu':
            y = F.relu(y)
        elif self.nonlinear in ['c2relu','smoothrelu']:
            y = c2relu(y)

        return y


    def extra_repr(self):
        s = ('{in_dim}, {out_dim}')

        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'

        return s.format(**self.__dict__)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels,  stride=1, padding=None,
            kernel_size=(3,3),  bn=True, nonlinear='relu',
            bias=True,
            **kwargs):
        """A 2d convolution block.  The convolution is followed by batch
        normalization.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride (int, optional): stride of the convolutions (default: 1)
            kernel_size (tuple, optional): kernel shape (default: 3)
            bn (bool, optional): turn on batch norm (default: True)
        """
        super().__init__()

        self.weight = nn.Parameter(th.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(th.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        self.stride = stride
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size=_pair(kernel_size)
        self.nonlinear=nonlinear

        if padding is None:
            self.padding = tuple([k//2 for k in kernel_size])
        else:
            self.padding = _pair(padding)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels, affine=False)
        else:
            self.bn = False


        self.reset_parameters()


    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):

        y = F.conv2d(x, self.weight, None, self.stride, self.padding, 1, 1)

        if self.bn:
            y = self.bn(y)

        if self.bias is not None:
            b = self.bias.view(1,self.out_channels,1,1)
            y = y+b

        if self.nonlinear=='relu':
            y = F.relu(y)
        elif self.nonlinear in ['c2relu','smoothrelu']:
            y = c2relu(y)

        return y


    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'
        return s.format(**self.__dict__)



class BasicBlock(nn.Module):

    def __init__(self, channels, kernel_size=(3,3), nonlinear='relu', **kwargs):
        """A basic 2d ResNet block [1].

        Args:
            channels: number of input and output channels
            kernel_size (tuple, optional): kernel shape (default: 3)

        [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2016.
            Deep Residual Learning for Image Recognition. arXiv:1512.03385
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = _pair(kernel_size)
        self.nonlinear=nonlinear

        self.conv0 = Conv(channels, channels, kernel_size=kernel_size, nonlinear=nonlinear)
        self.conv1 = Conv(channels, channels, kernel_size=kernel_size, nonlinear=False)

    def forward(self, x):

        y = self.conv0(x)
        y = self.conv1(y)

        y = (x+y)

        if self.nonlinear=='relu':
            y = F.relu(y)
        elif self.nonlinear in ['c2relu','smoothrelu']:
            y = c2relu(y)

        return y


    def extra_repr(self):
        s = ('{channels}, {channels}, kernel_size={kernel_size}')

        return s.format(**self.__dict__)

class Bottleneck(nn.Module):

    def __init__(self, channels, kernel_size=(3,3),  nonlinear='relu',
            **kwargs):
        """A basic 2d ResNet bottleneck block [1].

        Args:
            channels: number of input and output channels
            kernel_size (tuple, optional): kernel shape (default: 3)
            bn (bool, optional): turn on batch norm (default: False)

        [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2016.
            Deep Residual Learning for Image Recognition. arXiv:1512.03385
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = _pair(kernel_size)
        self.nonlinear=nonlinear

        self.conv0 = Conv(channels, channels//4, kernel_size=(1,1), nonlinear=nonlinear)
        self.conv1 = Conv(channels//4, channels//4, kernel_size=kernel_size, nonlinear=nonlinear)
        self.conv2 = Conv(channels//4, channels, kernel_size=(1,1), nonlinear=False)

    def forward(self, x):

        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)

        y = (x+y)

        if self.nonlinear=='relu':
            y = F.relu(y)
        elif self.nonlinear in ['c2relu','smoothrelu']:
            y = c2relu(y)

        return y


    def extra_repr(self):
        s = ('{channels}, {channels}, kernel_size={kernel_size}')

        return s.format(**self.__dict__)

class BranchBlock(nn.Module):

    def __init__(self, channels, kernel_size=(3,3), branches=2, nonlinear='relu',
            **kwargs):
        """A 2d ResNet block, where the channels are 'branched' into separate
        groups.  Every convolution is followed by batch normalization.

        This module is like a simplified ResNeXt block,
        without the 1x1 convolution layer aggregating the channels, using a
        mean to aggregate instead.

        Args:
            channels: number of input and output channels
            kernel_size (tuple, optional): kernel shape (default: 3)
            branches (int, optional): number of branches (default: 2)
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = _pair(kernel_size)
        self.branches = branches
        self.nonlinear=nonlinear

        padding = [k //2 for k in kernel_size]

        self.conv0 = nn.Conv2d(channels, channels*branches,
                                kernel_size, groups=1,
                                bias=False , padding=padding, stride=1)
        self.conv1 = nn.Conv2d(channels*branches, channels*branches,
                                kernel_size, groups=branches,
                                bias=False, padding=padding, stride=1)

        self.bn0 = nn.BatchNorm2d(channels * branches, affine=True)
        self.bn1 = nn.BatchNorm2d(channels * branches, affine=True)


    def forward(self, x):

        y = self.conv0(x)

        y = self.bn0(y)

        if self.nonlinear=='relu':
            y = F.relu(y)
        elif self.nonlinear in ['c2relu','smoothrelu']:
            y = c2relu(y)

        y = self.conv1(y)

        y = self.bn1(y)

        y = y.unsqueeze(1).chunk(self.branches,dim=2)
        y = th.cat(y, 1)

        y = y.mean(1)

        y = (x+y)

        if self.nonlinear=='relu':
            y = F.relu(y)
        elif self.nonlinear in ['c2relu','smoothrelu']:
            y = c2relu(y)

        return y

    def extra_repr(self):
        s = ('{channels}, {channels}, kernel_size={kernel_size}')

        return s.format(**self.__dict__)
