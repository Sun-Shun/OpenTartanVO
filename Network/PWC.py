# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu,  CMU
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

"""
Implementation of the PWC-DC network for optical flow estimation.
Based on the work by Sun et al., 2018 (PWC-Net) with densenet connections
and dilated convolutions.

Reference: Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz.
           PWC-Net: CNNs for optical flow using pyramid, warping, and cost volume.
           CVPR 2018.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .correlation import FunctionCorrelation


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class PWCDCNet(nn.Module):
    """
    PWC-DC net. Adds dilated convolutions and densenet connections to PWC-Net.
    """

    def __init__(self, md=4, flow_norm=20.0):
        """
        md  -- maximum displacement for cost-volume correlation (default: 4)
        flow_norm -- normalization factor applied to the flow output at test time
        """
        super(PWCDCNet, self).__init__()

        self.flow_norm = flow_norm

        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa = conv(32,  32, kernel_size=3, stride=1)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3aa = conv(64,  64, kernel_size=3, stride=1)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4aa = conv(96,  96, kernel_size=3, stride=1)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128, 128, kernel_size=3, stride=1)
        self.conv5b  = conv(128, 128, kernel_size=3, stride=1)
        self.conv6aa = conv(128, 196, kernel_size=3, stride=2)
        self.conv6a  = conv(196, 196, kernel_size=3, stride=1)
        self.conv6b  = conv(196, 196, kernel_size=3, stride=1)

        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd
        self.conv6_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow6 = predict_flow(od + dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 128 + 4
        self.conv5_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od + dd[4])
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 96 + 4
        self.conv4_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od + dd[4])
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 64 + 4
        self.conv3_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od + dd[4])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 32 + 4
        self.conv2_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,         128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,         128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,          96, kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,            64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,            32, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flo):
        """
        Warp feature map x according to flow flo.

        x:   [B, C, H, W]  (features from frame 2)
        flo: [B, 2, H, W]  (optical flow from frame 1 to frame 2)
        """
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo

        # Scale grid to [-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, align_corners=True)
        mask = torch.ones(x.size(), device=x.device)
        mask = F.grid_sample(mask, vgrid, align_corners=True)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def _multi_scale_conv(self, conv0, conv1, conv2, conv3, conv4, feat):
        x = torch.cat((conv0(feat), feat), 1)
        x = torch.cat((conv1(x), x), 1)
        x = torch.cat((conv2(x), x), 1)
        x = torch.cat((conv3(x), x), 1)
        x = torch.cat((conv4(x), x), 1)
        return x

    def _concat_two_levels(self, pred_fn, deconv_fn, upfeat_fn,
                           feat_high, feat_low1, feat_low2, scale):
        flow_high = pred_fn(feat_high)
        up_flow = deconv_fn(flow_high)
        up_feat = upfeat_fn(feat_high)
        warp_feat = self.warp(feat_low2, up_flow * scale)
        corr_low = FunctionCorrelation(tenFirst=feat_low1, tenSecond=warp_feat)
        corr_low = self.leakyRELU(corr_low)
        x = torch.cat((corr_low, feat_low1, up_flow, up_feat), 1)
        return x, flow_high

    def forward(self, x):
        """
        x: list/tuple of two RGB image tensors [B, 3, H, W]

        Returns (training):  (flow2, flow3, flow4, flow5, flow6)
                             all normalized by flow_norm, at scales 1/4, 1/8, 1/16, 1/32, 1/64
        Returns (eval):      flow2 / flow_norm  [B, 2, H/4, W/4]
        """
        im1 = x[0]
        im2 = x[1]

        # Encoder
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        # Coarsest level (level 6)
        corr6 = FunctionCorrelation(tenFirst=c16, tenSecond=c26)
        corr6 = self.leakyRELU(corr6)
        x6 = self._multi_scale_conv(self.conv6_0, self.conv6_1, self.conv6_2,
                                    self.conv6_3, self.conv6_4, corr6)
        x5, flow6 = self._concat_two_levels(self.predict_flow6, self.deconv6, self.upfeat6,
                                             x6, c15, c25, 0.625)

        x5 = self._multi_scale_conv(self.conv5_0, self.conv5_1, self.conv5_2,
                                    self.conv5_3, self.conv5_4, x5)
        x4, flow5 = self._concat_two_levels(self.predict_flow5, self.deconv5, self.upfeat5,
                                             x5, c14, c24, 1.25)

        x4 = self._multi_scale_conv(self.conv4_0, self.conv4_1, self.conv4_2,
                                    self.conv4_3, self.conv4_4, x4)
        x3, flow4 = self._concat_two_levels(self.predict_flow4, self.deconv4, self.upfeat4,
                                             x4, c13, c23, 2.5)

        x3 = self._multi_scale_conv(self.conv3_0, self.conv3_1, self.conv3_2,
                                    self.conv3_3, self.conv3_4, x3)
        x2, flow3 = self._concat_two_levels(self.predict_flow3, self.deconv3, self.upfeat3,
                                             x3, c12, c22, 5.0)

        x2 = self._multi_scale_conv(self.conv2_0, self.conv2_1, self.conv2_2,
                                    self.conv2_3, self.conv2_4, x2)
        flow2 = self.predict_flow2(x2)

        # Context network (dilated convolutions)
        x_dc = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x2))))
        refine = self.dc_conv7(self.dc_conv6(self.dc_conv5(x_dc)))
        flow2 = flow2 + refine

        if self.training:
            return flow2 / self.flow_norm, flow3 / self.flow_norm, \
                   flow4 / self.flow_norm, flow5 / self.flow_norm, \
                   flow6 / self.flow_norm
        else:
            return flow2 / self.flow_norm
