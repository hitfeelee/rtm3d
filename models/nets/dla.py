import numpy as np

import torch
from torch import nn
from torchvision.ops import DeformConv2d
from models.nets.ShapeSpec import ShapeSpec
from collections import OrderedDict
from utils import torch_utils
# -----------------------------------------------------------------------------
# DLA models
# -----------------------------------------------------------------------------

DLA_SPEC = {"34": {
    "levels": [1, 1, 1, 2, 2, 1],
    "channels": [16, 32, 64, 128, 256, 512],
    "block": "BasicBlock"
}
}


def create_model(config):
    arch = config.MODEL.BACKBONE
    spec = DLA_SPEC[arch.split('-')[-1]]
    model = DLABase(config,
                    levels=spec["levels"],
                    channels=spec["channels"],
                    block=eval(spec["block"]),
                    norm_func=nn.BatchNorm2d)
    return model


class DeformConv(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 norm_func):
        super(DeformConv, self).__init__()

        self.norm = norm_func(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.deform_conv = DeformConv2d(in_channels=in_channel,
                                        out_channels=out_channel,
                                        kernel_size=(3, 3),
                                        stride=1,
                                        padding=1,
                                        dilation=1)

    def forward(self, x):
        x = self.deform_conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_func,
                 stride=1,
                 dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias=False,
                               dilation=dilation)
        self.norm1 = norm_func(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=dilation,
                               bias=False,
                               dilation=dilation
                               )
        self.norm2 = norm_func(out_channels)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.relu(out)

        return out


class Tree(nn.Module):
    def __init__(self,
                 level,
                 block,
                 in_channels,
                 out_channels,
                 norm_func,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False
                 ):
        super(Tree, self).__init__()

        if root_dim == 0:
            root_dim = 2 * out_channels

        if level_root:
            root_dim += in_channels

        if level == 1:
            self.tree1 = block(in_channels,
                               out_channels,
                               norm_func,
                               stride,
                               dilation=dilation)

            self.tree2 = block(out_channels,
                               out_channels,
                               norm_func,
                               stride=1,
                               dilation=dilation)
        else:
            new_level = level - 1
            self.tree1 = Tree(new_level,
                              block,
                              in_channels,
                              out_channels,
                              norm_func,
                              stride,
                              root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation,
                              root_residual=root_residual)

            self.tree2 = Tree(new_level,
                              block,
                              out_channels,
                              out_channels,
                              norm_func,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation,
                              root_residual=root_residual)
        if level == 1:
            self.root = Root(root_dim,
                             out_channels,
                             norm_func,
                             root_kernel_size,
                             root_residual)

        self.level_root = level_root
        self.root_dim = root_dim
        self.level = level

        self.downsample = None
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)

        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          bias=False),

                norm_func(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        if children is None:
            children = []

        if self.downsample:
            bottom = self.downsample(x)
        else:
            bottom = x

        if self.project:
            residual = self.project(bottom)
        else:
            residual = bottom

        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)

        if self.level == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class Root(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_func,
                 kernel_size,
                 residual):
        super(Root, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=1,
                              bias=False,
                              padding=(kernel_size - 1) // 2)

        self.norm = norm_func(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.norm(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class DLABase(nn.Module):
    def __init__(self,
                 config,
                 levels,
                 channels,
                 block=BasicBlock,
                 residual_root=False,
                 norm_func=nn.BatchNorm2d,
                 ):
        super(DLABase, self).__init__()
        self.config = config
        self.channels = channels
        self.level_length = len(levels)
        self.kfns = self.config.MODEL.KFNs

        self.base_layer = nn.Sequential(nn.Conv2d(3,
                                                  channels[0],
                                                  kernel_size=7,
                                                  stride=1,
                                                  padding=3,
                                                  bias=False),

                                        norm_func(channels[0]),

                                        nn.ReLU(inplace=True))

        self.level0 = torch_utils.make_conv_level(in_channels=channels[0],
                                                  out_channels=channels[0],
                                                  num_convs=levels[0],
                                                  norm_func=norm_func)

        self.level1 = torch_utils.make_conv_level(in_channels=channels[0],
                                                  out_channels=channels[1],
                                                  num_convs=levels[0],
                                                  norm_func=norm_func,
                                                  stride=2)

        self.level2 = Tree(level=levels[2],
                           block=block,
                           in_channels=channels[1],
                           out_channels=channels[2],
                           norm_func=norm_func,
                           stride=2,
                           level_root=False,
                           root_residual=residual_root)

        self.level3 = Tree(level=levels[3],
                           block=block,
                           in_channels=channels[2],
                           out_channels=channels[3],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

        self.level4 = Tree(level=levels[4],
                           block=block,
                           in_channels=channels[3],
                           out_channels=channels[4],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

        self.level5 = Tree(level=levels[5],
                           block=block,
                           in_channels=channels[4],
                           out_channels=channels[5],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

        self._kfpn_spec = OrderedDict()
        for layer in self.kfns:
            i = int(layer[-1])
            self._kfpn_spec[layer] = ShapeSpec(channels=channels[i], stride=2**i)

    def forward(self, x):
        y = []
        x = self.base_layer(x)

        for i in range(self.level_length):
            layer = 'level{}'.format(i)
            x = getattr(self, layer)(x)
            if layer in self.kfns:
                y.append(x)

        return y

    def _make_conv_level(self, in_channels, out_channels, kernel_size=3, num_convs=1, norm_func=nn.BatchNorm2d,
                         stride=1, dilation=1):
        """
        make conv layers based on its number.
        """
        modules = []
        for i in range(num_convs):
            modules.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                norm_func(out_channels),
                nn.ReLU(inplace=True)])
            in_channels = out_channels

        return nn.Sequential(*modules)

    @property
    def kfpn_spec(self):
        return self._kfpn_spec

