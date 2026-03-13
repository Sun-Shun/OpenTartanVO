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
import os

import torch
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
from Datasets.VODataest import PoseDataset, FlowDataset, VODataset
from Datasets.utils import *
from Datasets.transformation import motion_ses2pose_quats
from evaluator.tartanair_evaluator import TartanAirEvaluator
import time
from Network.VONet import VONet
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

SEED = "12345"

class TartanVO(object):
    def __init__(self, args, train_img, train_flow, train_mask, train_pose, test_img, test_flow, test_mask, test_pose):
        self.args = args
        self.vonet = VONet()
        self.vonet.cuda()
        self.setup_seed(SEED)
        self.pose_std = np.array([0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=np.float32)
        # train types
        self.only_pose = self.args.only_pose
        self.only_flow = self.args.only_flow
        self.vo = self.args.vo
        # logger
        self.writer = SummaryWriter(log_dir=str(os.path.join(self.args.logs_dir, self.args.train_type)))
        self.Tensorboard_name = self.args.train_type
        # optimization parameter
        self.pose_model = self.args.pose_model
        self.flow_model = self.args.flow_model
        self.save_path = os.path.join(self.args.root_path, self.args.train_type)
        self.batch_size = self.args.batch_size
        self.num_workers = self.args.num_workers
        self.intrinsic = dataset_intrinsics(self.args.datastr)  # fx fy ox oy

        if self.only_pose:
            # data augment in gpu
            self.transform = Compose(
                [RandomResizeCrop((448, 640), self.intrinsic, max_scale=2.5, keep_center=True, fix_ratio=True, use_gpu=True),
                 DownscaleFlow()])
            self.transform_test = Compose([CropCenter((448, 640), self.intrinsic), DownscaleFlow()])

            self.trainDataset = PoseDataset(None, train_flow, None, train_pose, transform=self.transform)
            self.valid_Dataset = PoseDataset(None, test_flow, None, test_pose, transform=self.transform_test)

            self.train_dataloader = DataLoader(self.trainDataset, batch_size=self.batch_size, prefetch_factor=2,
                                               shuffle=True, num_workers=self.num_workers,
                                               drop_last=True, persistent_workers=True)
            self.valid_dataloader = DataLoader(self.valid_Dataset, batch_size=self.batch_size, prefetch_factor=2,
                                               shuffle=False, num_workers=self.num_workers,
                                               drop_last=False, persistent_workers=True)
        elif self.only_flow:
            # data augment must in cpu
            self.transform = Compose([Flow_RRCrop((448, 640), max_scale=1.0, keep_center=True, fix_ratio=False)])
            #self.transform = Compose([Flow_Crop((448, 640))])
            self.transform_test = Compose([Flow_Crop((448, 640))])

            self.trainDataset = FlowDataset(train_img, train_flow, None, None, transform=self.transform)
            self.valid_Dataset = FlowDataset(test_img, test_flow, None, None, transform=self.transform_test)

            self.train_dataloader = DataLoader(self.trainDataset, batch_size=self.batch_size, prefetch_factor=2,
                                               shuffle=True, num_workers=self.num_workers, drop_last=True)
            self.valid_dataloader = DataLoader(self.valid_Dataset, batch_size=self.batch_size, prefetch_factor=2,
                                               shuffle=False, num_workers=self.num_workers, drop_last=False)
        elif self.args.vo:
            # data augment in cpu
            self.transform = Compose([RandomResizeCrop((448, 640), self.intrinsic, max_scale=2.5, keep_center=True, fix_ratio=False, use_gpu=False)])
            self.transform_test = Compose([CropCenter((448, 640), self.intrinsic)])

            self.trainDataset = VODataset(train_img, train_flow, None, train_pose, transform=self.transform)
            self.valid_Dataset = VODataset(test_img, test_flow, None, test_pose, transform=self.transform_test)

            self.train_dataloader = DataLoader(self.trainDataset, batch_size=self.batch_size, prefetch_factor=2,
                                               shuffle=True, num_workers=self.num_workers, drop_last=True)
            self.valid_dataloader = DataLoader(self.valid_Dataset, batch_size=self.batch_size, prefetch_factor=2,
                                               shuffle=False, num_workers=self.num_workers, drop_last=False)

        self.test_lab_path = test_pose
        self.test_flow_file = test_flow
        self.test_img_path = test_img

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.deterministic = True

    def load_model(self, model, modelpath):
        preTrainDict = torch.load(modelpath)
        model_dict = model.state_dict()
        preTrainDictTemp = {k: v for k, v in preTrainDict.items() if k in model_dict}
        if 0 == len(preTrainDictTemp):
            print("Does not find any module to load. Try DataParallel version.")
            for k, v in preTrainDict.items():
                kk = k[7:]
                if kk in model_dict:
                    preTrainDictTemp[kk] = v
        if 0 == len(preTrainDictTemp):
            raise Exception("Could not load model from %s." % modelpath, "load_model")
        model_dict.update(preTrainDictTemp)
        model.load_state_dict(model_dict)
        print('Model loaded...')
        return model

    def pose_train(self, epochs, lr_rate, step_size, pre_train=True, weight_decay=0):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print('CUDA used.')
            self.vonet.cuda()
        if pre_train:
            self.load_model(self.vonet, self.pose_model)
        train_epochs = epochs
        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.vonet.flowPoseNet.parameters(), lr=lr_rate, betas=(0.9, 0.999),
                                      weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_size, gamma=0.2, last_epoch=-1)
        # Train
        for ep in range(train_epochs):
            self.vonet.train()
            st_t = time.time()
            print('=' * 50)
            count = 0
            loss_pose, loss_tran, loss_rot = 0.0, 0.0, 0.0
            for data in self.train_dataloader:
                count = count + 1
                flow = data['flow'].cuda(non_blocking=True)
                pose = data['pose'].cuda(non_blocking=True)
                intrinsic = data['intrinsic'].cuda(non_blocking=True)
                inputs = [flow, intrinsic, pose]
                loss, loss_tran, loss_rot = self.vonet.step_pose(inputs, optimizer)
                loss_pose += float(loss) / (len(self.train_dataloader))
                loss_tran += float(loss_tran) / (len(self.train_dataloader))
                loss_rot += float(loss_rot) / (len(self.train_dataloader))
                print('\rCurrent train batch: {}/{}'.format(count, len(self.train_dataloader)), end='')
            scheduler.step()
            print("\nTrain use time :{}".format(time.time() - st_t))
            # test
            time.sleep(0.003)
            self.vonet.eval()
            print('')
            count = 0
            v_loss_pose = 0
            for data in self.valid_dataloader:
                count = count + 1
                flow = data['flow'].cuda(non_blocking=True)
                pose = data['pose'].cuda(non_blocking=True)
                intrinsic = data['intrinsic'].cuda(non_blocking=True)
                inputs = [flow, intrinsic, pose]
                with torch.no_grad():
                    vls, _, _ = self.vonet.loss_pose(inputs)
                v_loss_pose += float(vls) / (len(self.valid_dataloader))
                print('\rCurrent vaild batch: {}/{}'.format(count, len(self.valid_dataloader)), end='')

            print('\nEpoch {} \ntrain loss mean: {}\nvalid loss mean: {}\n'.format(ep + 1, loss_pose, v_loss_pose))
            # logger loss
            logger = {'loss': loss_pose, 'loss_tran': loss_tran, 'loss_rot': loss_rot, 'vloss': v_loss_pose}
            self.writer.add_scalars(self.Tensorboard_name, logger, ep)
            # save model
            if ep + 1 == train_epochs or (ep + 1) % 5 == 0:
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                print('Save model at ep {}, mean of train loss: {}'.format(ep + 1, loss_pose))
                torch.save(self.vonet.flowPoseNet.state_dict(), os.path.join(self.save_path, str(ep + 1) + '.train'))
            torch.cuda.empty_cache()

    def flow_train(self, epochs, lr_rate, step_size, pre_train=True, weight_decay=0):
        scaler = GradScaler()
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print('CUDA used.')
            self.vonet.cuda()
        if pre_train:
            self.load_model(self.vonet.flowNet, self.flow_model)
        train_epochs = epochs
        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.vonet.flowNet.parameters(), lr=lr_rate, betas=(0.9, 0.999),
                                      weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_size, gamma=0.2, last_epoch=-1)
        # Train
        for ep in range(train_epochs):
            self.vonet.train()
            st_t = time.time()
            print('=' * 50)
            count = 0
            loss_flow = 0
            for data in self.train_dataloader:
                count = count + 1
                img1 = data['img1'].cuda(non_blocking=True)
                img2 = data['img2'].cuda(non_blocking=True)
                flow = data['flow'].cuda(non_blocking=True)
                mask = data['mask'].cuda(non_blocking=True) if 'mask' in data else None
                inputs = [img1, img2, flow, mask]
                loss = self.vonet.step_flow(inputs, optimizer, scaler)
                loss_flow += float(loss) / (len(self.train_dataloader))
                print('\rCurrent train batch: {}/{}'.format(count, len(self.train_dataloader)), end='')
            scheduler.step()
            print("\nTrain use time :{}".format(time.time() - st_t))
            # test
            time.sleep(0.003)
            self.vonet.eval()
            print('')
            count = 0
            v_loss_flow = 0
            for data in self.valid_dataloader:
                count = count + 1
                img1 = data['img1'].cuda(non_blocking=True)
                img2 = data['img2'].cuda(non_blocking=True)
                flow = data['flow'].cuda(non_blocking=True)
                mask = data['mask'].cuda(non_blocking=True) if 'mask' in data else None
                inputs = [img1, img2, flow, mask]
                with torch.no_grad():
                    vls = self.vonet.loss_flow(inputs)
                v_loss_flow += float(vls)
                print('\rCurrent vaild batch: {}/{}'.format(count, len(self.valid_dataloader)), end='')
            v_loss_flow /= (len(self.valid_dataloader))
            print('\nEpoch {} \ntrain loss mean: {}\nvalid loss mean: {}\n'.format(ep + 1, loss_flow, v_loss_flow))
            # logger loss
            logger = {'loss': loss_flow, 'vloss': v_loss_flow}
            self.writer.add_scalars(self.Tensorboard_name, logger, ep)
            # save model
            if ep + 1 == train_epochs or (ep + 1) % 5 == 0:
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                print('Save model at ep {}, mean of train loss: {}'.format(ep + 1, loss_flow))
                torch.save(self.vonet.flowNet.state_dict(), os.path.join(self.save_path, str(ep + 1) + '.train'))

    def vo_train(self, epochs, lr_rate, step_size, pre_train=True, weight_decay=0):
        scaler = GradScaler()
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print('CUDA used.')
            self.vonet.cuda()
        if pre_train:
            self.load_model(self.vonet.flowNet, self.flow_model)
            self.load_model(self.vonet.flowPoseNet, self.pose_model)

        train_epochs = epochs
        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.vonet.parameters(), lr=lr_rate, betas=(0.9, 0.999),
                                      weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_size, gamma=0.2, last_epoch=-1)
        # Train
        for ep in range(train_epochs):
            self.vonet.train()
            st_t = time.time()
            print('=' * 50)
            count = 0
            loss, loss_flow, loss_pose = 0.0, 0.0, 0.0
            for data in self.train_dataloader:
                count = count + 1
                img1 = data['img1'].cuda(non_blocking=True)
                img2 = data['img2'].cuda(non_blocking=True)
                gt_flow = data['flow'].cuda(non_blocking=True)
                pose = data['pose'].cuda(non_blocking=True)
                intrinsic = data['intrinsic'].cuda(non_blocking=True)
                mask = data['mask'].cuda(non_blocking=True) if 'mask' in data else None
                # normalize  in this, reduce CPU overhead of loading datas
                gt_flow = F.interpolate(gt_flow, scale_factor=0.25, mode='bilinear')
                intrinsic = F.interpolate(intrinsic, scale_factor=0.25, mode='bilinear')
                mask = F.interpolate(mask, scale_factor=0.25, mode='bilinear') if (mask is not None) else None
                # load datas to model
                inputs = [img1, img2, intrinsic, gt_flow, pose, mask]
                loss, loss_flow, loss_pose = self.vonet.step_vo(inputs, optimizer, scaler)
                loss += float(loss) / (len(self.train_dataloader))
                loss_flow += float(loss_flow) / (len(self.train_dataloader))
                loss_pose += float(loss_pose) / (len(self.train_dataloader))
                print('\rCurrent train batch: {}/{}'.format(count, len(self.train_dataloader)), end='')

            scheduler.step()
            print("\nTrain use time :{}".format(time.time() - st_t))
            # test
            time.sleep(0.003)
            self.vonet.eval()
            print('')
            count = 0
            vloss, vloss_flow, vloss_pose = 0.0, 0.0, 0.0
            for data in self.valid_dataloader:
                count = count + 1
                img1 = data['img1'].cuda(non_blocking=True)
                img2 = data['img2'].cuda(non_blocking=True)
                gt_flow = data['flow'].cuda(non_blocking=True)
                pose = data['pose'].cuda(non_blocking=True)
                intrinsic = data['intrinsic'].cuda(non_blocking=True)
                mask = data['mask'].cuda(non_blocking=True) if 'mask' in data else None
                # normalize  in this, reduce CPU overhead of loading datas
                gt_flow = F.interpolate(gt_flow, scale_factor=0.25, mode='bilinear')
                intrinsic = F.interpolate(intrinsic, scale_factor=0.25, mode='bilinear')
                mask = F.interpolate(mask, scale_factor=0.25, mode='bilinear') if (mask is not None) else None
                inputs = [img1, img2, intrinsic, gt_flow, pose, mask]
                with torch.no_grad():
                    vls, vls_flow, vls_pose = self.vonet.loss_vo(inputs)
                vloss += float(vls) / (len(self.valid_dataloader))
                vloss_flow += float(vls_flow) / (len(self.valid_dataloader))
                vloss_pose += float(vls_pose) / (len(self.valid_dataloader))
                print('\rCurrent vaild batch: {}/{}'.format(count, len(self.valid_dataloader)), end='')

            print('\nEpoch {} \ntrain loss mean: {}\nvalid loss mean: {}\n'.format(ep + 1, loss, vloss))
            # logger loss
            logger = {'loss': loss, 'loss_flow': loss_flow, 'loss_pose': loss_pose, 'vloss': vloss,
                      'vloss_flow': vloss_flow, 'vloss_pose': vloss_pose}
            self.writer.add_scalars(self.Tensorboard_name, logger, ep)
            # save model
            if ep + 1 == train_epochs or (ep + 1) % 5 == 0:
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                print('Save model at ep {}, mean of train loss: {}'.format(ep + 1, loss))
                torch.save(self.vonet.state_dict(), os.path.join(self.save_path, str(ep + 1) + '.train'))

    def test_batch(self, model_path, imgs_path=None, flows_path=None, poses_path=None, test_type=None):

        for i in range(len(self.test_lab_path)):
            test_img, test_flow, test_pose = [], [], []
            if test_type == "flow":
                test_flow.append(imgs_path[i])
                test_pose.append(flows_path[i])
                flow_dataset = PoseDataset(None, test_flow, test_pose, transform=self.transform_test)
                flow_dataLoader = DataLoader(flow_dataset, batch_size=1, shuffle=False, num_workers=5, drop_last=False)
            elif test_type == "pose":
                test_flow.append(flows_path[i])
                test_pose.append(poses_path[i])
                pose_dataset = VODataset(None, test_flow, test_pose, transform=self.transform_test)
                pose_dataLoader = DataLoader(pose_dataset, batch_size=1, shuffle=False, num_workers=5, drop_last=False)
            elif test_type == "vo":
                test_img.append(imgs_path[i])
                test_flow.append(flows_path[i])
                test_pose.append(poses_path[i])
                vo_dataset = VODataset(test_img, test_flow, test_pose, transform=self.transform_test)
                vo_dataLoader = DataLoader(vo_dataset, batch_size=1, shuffle=False, num_workers=5, drop_last=False)

            if test_type == "flow":
                count = 0
                epe = 0.0
                self.load_model(self.vonet, model_path)
                self.vonet.eval()
                for datas in flow_dataLoader:
                    count = count + 1
                    img1 = datas['img1'].cuda(non_blocking=True)
                    img2 = datas['img2'].cuda(non_blocking=True)
                    flow = datas['flow'].cuda(non_blocking=True)
                    tes_inputs = [img1, img2]
                    with torch.no_grad():
                        pre_flows = self.vonet.forward(tes_inputs, only_flow=True)
                    pre_flow = pre_flows[-1].cpu().numpy()
                    gt_flow = flow.detach().cpu().numpy()
                    epe += torch.sum((pre_flow - gt_flow) ** 2, dim=0).sqrt()
                    print('\rCurrent test batch: {}/{}'.format(count, len(flow_dataLoader)), end='')
                print("\nEPE:{}".format(epe/len(flow_dataLoader)))

            elif test_type == "pose":
                motionlist = []
                count = 0
                self.load_model(self.vonet, model_path)
                self.vonet.eval()
                for datas in pose_dataLoader:
                    count = count + 1
                    flow = datas['flow'].cuda(non_blocking=True)
                    intrinsic = datas['intrinsic'].cuda(non_blocking=True)
                    gt_pose = datas['pose'].cuda(non_blocking=True)
                    tes_inputs = [flow, intrinsic]
                    with torch.no_grad():
                        pose = self.vonet.forward(tes_inputs, only_pose=True)
                    posenp = pose.data.cpu().numpy()
                    posenp = posenp * self.pose_std  # The output is normalized during training, now scale it back
                    motions_gt = gt_pose.cpu()
                    scale = np.linalg.norm(motions_gt[:, :3], axis=1)
                    trans_est = posenp[:, :3]
                    trans_est = trans_est / np.linalg.norm(trans_est, axis=1).reshape(-1, 1) * scale.reshape(-1, 1)
                    posenp[:, :3] = trans_est
                    motions = posenp
                    motionlist.extend(motions)
                    print('\rCurrent test batch: {}/{}'.format(count, len(pose_dataLoader)), end='')
                # 得到绝对位姿的数据 （x y z 四元数）
                estposes = motion_ses2pose_quats(np.array(motionlist))
                evaluator = TartanAirEvaluator()
                results = evaluator.evaluate_one_trajectory(test_pose, estposes, scale=False, kittitype=False)
                seq_name = 'results/' + 'test_' + test_flow[0][-11:] + '.png'

                plot_traj(results['gt_aligned'], results['est_aligned'], savefigname=seq_name,
                          title='ATE %.4f' % (results['ate_score']))
                # np.savetxt('results/' + 'esti_' + seq_name + '.txt', results['est_aligned'])
                print('\r ATE loss: {}'.format(results['ate_score']))
                print("==> ATE: %.4f" % (results['ate_score']))

            elif test_type == "vo":
                motionlist = []
                count = 0
                self.load_model(self.vonet, model_path)
                self.vonet.eval()
                for datas in vo_dataLoader:
                    count = count + 1
                    img1 = datas['img1'].cuda(non_blocking=True)
                    img2 = datas['img2'].cuda(non_blocking=True)
                    intrinsic = datas['intrinsic'].cuda(non_blocking=True)
                    gt_pose = datas['pose'].cuda(non_blocking=True)
                    tes_inputs = [img1, img2, intrinsic]
                    with torch.no_grad():
                        _, pose = self.vonet.forward(tes_inputs)
                    posenp = pose.data.cpu().numpy()
                    posenp = posenp * self.pose_std  # The output is normalized during training, now scale it back
                    motions_gt = gt_pose.cpu()
                    scale = np.linalg.norm(motions_gt[:, :3], axis=1)
                    trans_est = posenp[:, :3]
                    trans_est = trans_est / np.linalg.norm(trans_est, axis=1).reshape(-1, 1) * scale.reshape(-1, 1)
                    posenp[:, :3] = trans_est
                    motions = posenp
                    motionlist.extend(motions)
                    print('\rCurrent test batch: {}/{}'.format(count, len(vo_dataLoader)), end='')

                # 得到绝对位姿的数据 （x y z 四元数）
                estposes = motion_ses2pose_quats(np.array(motionlist))
                evaluator = TartanAirEvaluator()
                results = evaluator.evaluate_one_trajectory(test_pose, estposes, scale=False, kittitype=False)
                seq_name = 'results/' + 'test_' + test_flow[0][-11:] + '.png'
                plot_traj(results['gt_aligned'], results['est_aligned'], savefigname=seq_name,title='ATE %.4f' % (results['ate_score']))
                # np.savetxt('results/' + 'esti_' + seq_name + '.txt', results['est_aligned'])
                print('\r ATE loss: {}'.format(results['ate_score']))
                print("==> ATE: %.4f" % (results['ate_score']))

