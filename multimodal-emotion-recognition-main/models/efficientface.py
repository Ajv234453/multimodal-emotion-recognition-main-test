# -*- coding: utf-8 -*-
"""
This code is modified from https://github.com/zengqunzhao/EfficientFace/blob/master/models/EfficientFace.py
"""

import torch
import torch.nn as nn

# def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
#     #return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)
#     layers = []
#     layers.append(
#         nn.Conv2d(
#             in_channels=i,
#             out_channels=i,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             bias=bias
#         )
#     )
#
#     layers.append(
#         nn.Conv2d(
#             in_channels=i,
#             out_channels=o,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=0
#         )
#     )
#     return nn.Sequential(*layers)

import torch.nn as nn


#这是一个实现了深度可分离卷积+残差连接的模块，省略了激活函数以轻量级
class DepthwiseResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        # 主路径的深度可分离卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, padding=0, bias=bias
        )

        # 残差路径处理
        self.use_shortcut = (
                in_channels == out_channels and
                stride == 1 and
                padding == (kernel_size - 1) // 2
        )

        if not self.use_shortcut:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=stride,
                padding=0, bias=bias
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out + residual


def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
    return DepthwiseResidual(i, o, kernel_size, stride, padding, bias)


import torch
import torch.nn as nn
import torch.nn.functional as F


class DRWS(nn.Module):
    def __init__(self, i, o, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()

        # 动态卷积部分
        self.conv_branches = nn.ModuleList([
            depthwise_conv(i, o, kernel_size, stride, padding, bias)
            for _ in range(4)
        ])
        self.alphas = nn.Parameter(torch.randn(4))  # 初始化等权重

        # 残差路径处理
        self.need_shortcut = (i != o) or (stride != 1)
        if self.need_shortcut:
            self.shortcut = nn.Conv2d(
                i, o, kernel_size=1,
                stride=stride, bias=bias
            )
        else:
            self.shortcut = nn.Identity()

        # 主路径处理层
        self.bn = nn.BatchNorm2d(o)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 动态卷积加权和
        weights = F.softmax(self.alphas, dim=0)
        dynamic_out = sum(w * branch(x) for w, branch in zip(weights, self.conv_branches))

        # 主路径处理：BN → ReLU
        main_out = self.relu(self.bn(dynamic_out))

        # 残差连接
        residual = self.shortcut(x)

        # 最终输出
        return self.relu(main_out + residual)


def DRWS_Conv(i, o, kernel_size, stride=1, padding=0, bias=False):
    return DRWS(i, o, kernel_size, stride, padding, bias)


#groups=i：每个输入通道都有一个独立的卷积滤波器，实现了深度卷积。
#普通卷积，每个卷积核得到一个map；深度卷积，每个通道得到一个map，再对得到的map与1*1的核进行点积
#深度可分离卷积参数数量和运算成本更低

#通道打乱


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups #向下取整
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class LocalFeatureExtractor(nn.Module):

    def __init__(self, inplanes, planes, index):
        super(LocalFeatureExtractor, self).__init__()
        self.index = index

        norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()

        self.conv1_1 = DRWS(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn1_1 = norm_layer(planes)
        self.conv1_2 = DRWS(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = norm_layer(planes)

        self.conv2_1 = DRWS(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = norm_layer(planes)
        self.conv2_2 = DRWS(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = norm_layer(planes)

        self.conv3_1 = DRWS(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn3_1 = norm_layer(planes)
        self.conv3_2 = DRWS(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = norm_layer(planes)

        self.conv4_1 = DRWS(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn4_1 = norm_layer(planes)
        self.conv4_2 = DRWS(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = norm_layer(planes)

    def forward(self, x):

        patch_11 = x[:, :, 0:28, 0:28]
        patch_21 = x[:, :, 28:56, 0:28]
        patch_12 = x[:, :, 0:28, 28:56]
        patch_22 = x[:, :, 28:56, 28:56]

        out_1 = self.conv1_1(patch_11)
        #批量归一化
        out_1 = self.bn1_1(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.conv1_2(out_1)
        out_1 = self.bn1_2(out_1)
        out_1 = self.relu(out_1)

        out_2 = self.conv2_1(patch_21)
        out_2 = self.bn2_1(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.conv2_2(out_2)
        out_2 = self.bn2_2(out_2)
        out_2 = self.relu(out_2)

        out_3 = self.conv3_1(patch_12)
        out_3 = self.bn3_1(out_3)
        out_3 = self.relu(out_3)
        out_3 = self.conv3_2(out_3)
        out_3 = self.bn3_2(out_3)
        out_3 = self.relu(out_3)

        out_4 = self.conv4_1(patch_22)
        out_4 = self.bn4_1(out_4)
        out_4 = self.relu(out_4)
        out_4 = self.conv4_2(out_4)
        out_4 = self.bn4_2(out_4)
        out_4 = self.relu(out_4)

        out1 = torch.cat([out_1, out_2], dim=2)
        out2 = torch.cat([out_3, out_4], dim=2)
        out = torch.cat([out1, out2], dim=3)

        return out


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))
        #不论步长，主分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True))

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        #branch2和branch1是nn.Sequential对象，会依次调用其中的所有层
        return out
