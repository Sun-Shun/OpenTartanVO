# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import torch
import torch.nn as nn
import torch.nn.functional as F
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

def conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bn_layer=False, bias=True):
    if bn_layer:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride,
                      dilation=dilation),
            nn.ReLU(inplace=True)
        )


def linear(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes),
        nn.ReLU(inplace=True)
    )


def linear_no(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = conv(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, pad, dilation)
        self.grn = GRN(planes)
        self.downsample = downsample
        self.downsample_conv = nn.Sequential(
            LayerNorm(inplanes, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(inplanes, planes, kernel_size=2, stride=2),
        )
        self.stride = stride

    def forward(self, x):
        if self.stride != 1:
            out = self.downsample_conv(x)
        else:
            out = self.conv1(x)
        out = self.conv2(out)
        out = F.relu(out)
        out = out.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        out = self.grn(out)
        out = out.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return out


class VOFlowRes(nn.Module):
    def __init__(self):
        super(VOFlowRes, self).__init__()
        inputnum = 4
        blocknums = [2, 2, 3, 4, 6, 7, 3]
        outputnums = [32, 64, 64, 128, 128, 256, 256]
        # outputnums = [32, 64, 64, 128, 256, 512, 1024]

        # self.hard_mask = hard_mask()
        self.firstconv = nn.Sequential(conv(inputnum, 32, 3, 2, 1, 1, False),
                                       conv(32, 32, 3, 1, 1, 1),
                                       conv(32, 32, 3, 1, 1, 1))
        self.inplanes = 32
        self.layer1 = self._make_layer(BasicBlock, outputnums[2], blocknums[2], 2, 1, 1)  # 40 x 28
        self.layer2 = self._make_layer(BasicBlock, outputnums[3], blocknums[3], 2, 1, 1)  # 20 x 14
        self.layer3 = self._make_layer(BasicBlock, outputnums[4], blocknums[4], 2, 1, 1)  # 10 x 7
        self.layer4 = self._make_layer(BasicBlock, outputnums[5], blocknums[5], 2, 1, 1)  # 5 x 4
        self.layer5 = self._make_layer(BasicBlock, outputnums[6], blocknums[6], 2, 1, 1)  # 3 x 2

        fcnum = outputnums[6] * 2

        fc1_trans = linear_no(fcnum, 128)
        fc2_trans = linear(128, 32)
        fc3_trans = nn.Linear(32, 3)

        fc1_rot = linear_no(fcnum, 128)
        fc2_rot = linear(128, 32)
        fc3_rot = nn.Linear(32, 3)

        self.voflow_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        self.voflow_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
            #                        kernel_size=1, stride=stride)
            downsample = nn.Sequential(
                LayerNorm(self.inplanes, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=2, stride=2),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):  # x:[1,6,H,W]

        x = self.firstconv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x)

        x = x.view(x.shape[0], -1)
        x_trans = self.voflow_trans(x)
        x_rot = self.voflow_rot(x)

        return torch.cat((x_trans, x_rot), dim=1)
