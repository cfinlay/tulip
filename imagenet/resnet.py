import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Function



__all__ = ['ResNet', 'resnet18', 'resnet34', 'smoothresnet50', 'resnet50', 'resnet101',
           'resnet152']


#TODO: this should be written in C++ and CUDA
class C2ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = F.relu(x+0.5,inplace=True)
        y = F.relu(x,inplace=True)
        return torch.max(-0.5*z.pow(4) + z.pow(3), 
                      y)




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,nonlinearity='relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.relu = C2ReLU() if nonlinearity=='c2relu' else nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,nonlinearity='relu'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, affine=True)
        self.relu = C2ReLU() if nonlinearity=='c2relu' else nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,nonlinearity='relu'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.relu = C2ReLU() if nonlinearity=='c2relu' else nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], nonlinearity=nonlinearity)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, nonlinearity=nonlinearity)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, nonlinearity=nonlinearity)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, nonlinearity=nonlinearity)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride=1,nonlinearity='relu'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,nonlinearity=nonlinearity))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,nonlinearity=nonlinearity))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def init_dist_weights(model):
    # https://arxiv.org/pdf/1706.02677.pdf
    # https://github.com/pytorch/examples/pull/262
    for m in model.modules():
        if isinstance(m, BasicBlock): m.bn2.weight = nn.Parameter(torch.zeros_like(m.bn2.weight))
        if isinstance(m, Bottleneck): m.bn3.weight = nn.Parameter(torch.zeros_like(m.bn3.weight))
        if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(bn0=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if bn0: init_dist_weights(model)
    return model

def smoothresnet50(bn0=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], nonlinearity='c2relu', **kwargs)
    if bn0: init_dist_weights(model)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
