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
import torch
import torch.nn as nn
from .RAFT.raft import RAFT as FlowNet
from .VOFlowNet import VOFlowRes as FlowPoseNet
from Datasets.utils import *
import itertools
autocast = torch.cuda.amp.autocast

class VONet(nn.Module):
    def __init__(self):
        super(VONet, self).__init__()
        self.flowNet = FlowNet(mixed_precision=False)
        self.flowPoseNet = FlowPoseNet()
        self.criterion = nn.L1Loss()


    def forward(self, x, only_pose=False,only_flow=False):
        if only_pose:
            intrinsic_flow = torch.cat((x[0], x[1]), dim=1)
            pose = self.flowPoseNet(intrinsic_flow)
            return pose
        elif only_flow:
            flow = self.flowNet(x[0:2])
            return flow
        else:
            flow = self.flowNet(x[0:2])[-1]
            flow = F.interpolate(flow, scale_factor=0.25, mode='bilinear')
            intrinsic_flow = torch.cat((flow, x[2]), dim=1)
            with autocast(enabled=False):
                pose = self.flowPoseNet(intrinsic_flow)
            return flow, pose


    def loss_pose(self, x):
        '''
        x: [flow,intrinsic,pose]
        y: pose
        '''
        gt_pose = x[2]
        pose = self.forward(x, only_pose=True)
        pose_trans = pose[:, :3]
        pose_rot = pose[:, 3:]
        tran_norm = pose_trans / torch.norm(pose_trans, dim=1).view(-1, 1)
        loss_tran = self.criterion(tran_norm, gt_pose[:, :3])  # 前三列为平移误差
        loss_rot = self.criterion(pose_rot, gt_pose[:, 3:])  # 后三列为角度误差
        loss = (loss_tran + loss_rot) / 2.0
        return loss, loss_tran, loss_rot

    def loss_flow(self, x, gamma=0.85):
        '''
        x: [img1,img2,flow,mask]
        '''
        flow_loss = 0.0
        max_flow = 400
        flow_preds = self.forward(x, only_flow=True)
        gt_flow = x[2]
        bath, _, height, width = gt_flow.size()
        mask_flow = x[-1] if (x[-1] is not None) else torch.zeros((bath, 1, height, width)).cuda()
        mask_flow = 1-(mask_flow >= 0.5).int()
        # mask pixels and extremely large diplacements
        mag = torch.sum(gt_flow ** 2, dim=1).sqrt()
        valid = (mask_flow.squeeze(1)) & (mag < max_flow)

        for i in range(len(flow_preds)):
            i_weight = gamma ** (len(flow_preds) - i - 1)
            i_loss = (flow_preds[i] - gt_flow).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        return flow_loss

    def loss_vo(self, x):
        '''
        x: [img1,img2,intrinsic,flow,pose,mask]
        '''
        max_flow = 400
        gt_flow, gt_pose = x[3], x[4]
        bath, _, height, width = gt_flow.size()
        flow, pose = self.forward(x)
        mask_flow = x[-1] if (x[-1] is not None) else torch.zeros((bath, 1, height, width)).cuda()
        mask_flow = 1-(mask_flow >= 0.5).int()
        mag = torch.sum(gt_flow ** 2, dim=1).sqrt()
        valid = (mask_flow.squeeze(1)) & (mag < max_flow)
        # FlowNet
        loss_flow = torch.abs((flow-gt_flow)*valid[:, None]).mean()
        # PoseNet
        pose_trans = pose[:, :3]
        pose_rot = pose[:, 3:]
        tran_norm = pose_trans / torch.norm(gt_pose, dim=1).view(-1, 1)
        trans_loss = self.criterion(tran_norm, gt_pose[:, :3])  # 前三列为平移误差
        rot_loss = self.criterion(pose_rot, gt_pose[:, 3:])  # 后三列为角度误差
        loss_pose = (rot_loss + trans_loss) / 2.0
        loss_vo = 0.1*loss_flow + loss_pose

        return loss_vo, loss_flow, loss_pose

    def step_pose(self, x, optimizer):
        '''
        x: [flow,intrinsic,pose,mask]
        '''
        optimizer.zero_grad()
        loss, loss_tran, loss_rot = self.loss_pose(x)
        loss.backward()
        optimizer.step()
        return loss.item(), loss_tran.item(), loss_rot.item()

    def step_flow(self, x, optimizer,scaler):
        '''
        x: [img1,img2,flow,mask]
        '''
        optimizer.zero_grad()

        loss_flow = self.loss_flow(x)
        # scaler.scale(loss_flow).backward()
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(self.flowNet.parameters(),1.0)
        # scaler.step(optimizer)
        # scaler.update()
        loss_flow.backward()
        optimizer.step()
        return loss_flow.item()

    def  step_vo(self, x, optimizer, scaler):
        '''
        x: [img1,img2,intrinsic,flow,pose,mask]
        y: [flow,pose]
        '''
        optimizer.zero_grad()
        parameters = itertools.chain(self.flowNet.parameters(), self.flowPoseNet.parameters())
        loss, loss_flow, loss_pose = self.loss_vo(x)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters,1.0)
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        return loss.item(), loss_flow.item(), loss_pose.item()

    # def L1_new(self, est_data, gt_data, un_vo, is_trans):
    #     loss = 0.0
    #     un_vo_fine = torch.log(1+torch.exp(un_vo))+1e-6
    #     for i in range(3):
    #         if is_trans is False:
    #             loss_temp1 = (((gt_data[:, i] - est_data[:, i]) ** 2) / un_vo_fine[:, i]) * 0.5
    #             log_un = un_vo_fine[:, i+3]+1
    #             loss += (loss_temp1 + 0.5 * torch.log(log_un))/3.0
    #         else:
    #             loss_temp1 = (((gt_data[:, i] - est_data[:, i]) ** 2) / un_vo_fine[:, i]) * 0.5
    #             log_un = un_vo_fine[:, i]+1
    #             loss += (loss_temp1 + 0.5 * torch.log(log_un)) / 3.0
    #     loss = torch.mean(loss)
    #     return loss
