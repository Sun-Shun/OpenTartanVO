import os
from TartanVO import TartanVO
import argparse
from glob import glob
from Network.VONet import VONet
from Datasets.VODataest import PoseDataset, FlowDataset, VODataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from Datasets.transformation import motion_ses2pose_quats
from evaluator.tartanair_evaluator import TartanAirEvaluator
from Datasets.utils import *
import multiprocessing
# ###############################相关路径设置######################################
test_scene = glob('/home/zhang/Data_new/data/tartanair/test')
test_scene.sort()
# dataset root directory
test_img, test_flow, test_pose = [], [], []
for v_scene in test_scene:
    test_img_dir = glob(v_scene + "/test_img" + "/*")
    test_flow_dir = glob(v_scene + "/test_flow" + "/*")
    test_pose_dir = glob(v_scene + "/test_pose" + "/*")
    try:
        assert len(test_flow_dir) == len(test_pose_dir)
    except AssertionError:
        print(v_scene)

    test_img_dir.sort()
    test_flow_dir.sort()
    test_pose_dir.sort()
    test_img += test_img_dir
    test_flow += test_flow_dir
    test_pose += test_pose_dir

def VO_test(model, model_path,datastr, imgs_path=None, flows_path=None, poses_path=None, test_type=None):
    vonet = model
    pose_std = np.array([0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=np.float32)
    dict_intrinsic = dataset_intrinsics(datastr)
    if test_type == "flow":
        load_model(vonet.flowNet, model_path)
    elif test_type == "pose":
        multiprocessing.set_start_method('spawn')
        load_model(vonet.flowPoseNet, model_path)
    elif test_type == "vo":
        load_model(vonet, model_path)

    for i in range(len(flows_path)):
        test_imgs, test_flows, test_poses = [], [], []
        if test_type == "flow":
            test_imgs.append(imgs_path[i])
            test_flows.append(flows_path[i])
            transform_test = Compose([Flow_Crop((448, 640))])
            flow_dataset = FlowDataset(test_imgs, test_flows, None, None, transform=transform_test)
            flow_dataLoader = DataLoader(flow_dataset, batch_size=1, shuffle=False, num_workers=5, drop_last=False)
        elif test_type == "pose":
            test_flows.append(flows_path[i])
            test_poses.append(poses_path[i])
            transform_test = Compose([CropCenter((448, 640), dict_intrinsic), DownscaleFlow()])
            pose_dataset = PoseDataset(None, test_flows, None, test_poses, transform=transform_test, is_test=True)
            pose_dataLoader = DataLoader(pose_dataset, batch_size=1, shuffle=False, num_workers=5, drop_last=False)
        elif test_type == "vo":
            test_imgs.append(imgs_path[i])
            test_flows.append(flows_path[i])
            test_poses.append(poses_path[i])
            transform_test = Compose([CropCenter((448, 640), dict_intrinsic), DownscaleFlow()])
            vo_dataset = VODataset(test_imgs, test_flows, None, test_poses, transform=transform_test, is_test=True)
            vo_dataLoader = DataLoader(vo_dataset, batch_size=1, shuffle=False, num_workers=5, drop_last=False)

        if test_type == "flow":
            count = 0
            epe = 0.0
            vonet.eval()
            for datas in flow_dataLoader:
                count = count + 1
                img1 = datas['img1'].cuda(non_blocking=True)
                img2 = datas['img2'].cuda(non_blocking=True)
                flow = datas['flow'].cuda(non_blocking=True)
                tes_inputs = [img1, img2]
                with torch.no_grad():
                    pre_flows = vonet.forward(tes_inputs, only_flow=True)
                pre_flow = pre_flows[-1].cpu()
                gt_flow = flow.detach().cpu()
                epe += torch.mean((pre_flow - gt_flow) ** 2).sqrt().numpy()
                print('\rCurrent test batch: {}/{}'.format(count, len(flow_dataLoader)), end='')
            print("\nEPE:{}".format(epe/len(flow_dataLoader)))

        elif test_type == "pose":
            motionlist = []
            count = 0
            vonet.eval()
            for datas in pose_dataLoader:
                count = count + 1
                flow = datas['flow'].cuda(non_blocking=True)
                intrinsic = datas['intrinsic'].cuda(non_blocking=True)
                gt_pose = datas['pose'].cuda(non_blocking=True)
                tes_inputs = [flow, intrinsic]
                with torch.no_grad():
                    pose = vonet.forward(tes_inputs, only_pose=True)
                posenp = pose.data.cpu().numpy()
                posenp = posenp * pose_std  # The output is normalized during training, now scale it back
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
            results = evaluator.evaluate_one_trajectory(test_poses, estposes, scale=False, kittitype=False)
            seq_name = 'results/' + 'test_' + test_flow[0].split('/')[-1] + '.png'
            plot_traj(results['gt_aligned'], results['est_aligned'], savefigname=seq_name,title='ATE %.4f' % (results['ate_score']))
            # np.savetxt('results/' + 'esti_' + seq_name + '.txt', results['est_aligned'])
            print('\r ATE loss: {}'.format(results['ate_score']))
            print("==> ATE: %.4f" % (results['ate_score']))

        elif test_type == "vo":
            motionlist = []
            count = 0
            vonet.eval()
            for datas in vo_dataLoader:
                count = count + 1
                img1 = datas['img1'].cuda(non_blocking=True)
                img2 = datas['img2'].cuda(non_blocking=True)
                intrinsic = datas['intrinsic'].cuda(non_blocking=True)
                flow = datas['flow'].cuda(non_blocking=True)
                gt_pose = datas['pose'].cuda(non_blocking=True)
                tes_inputs = [img1, img2,flow, intrinsic]
                with torch.no_grad():
                    _, pose = vonet.forward(tes_inputs)
                posenp = pose.data.cpu().numpy()
                posenp = posenp * pose_std  # The output is normalized during training, now scale it back
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
            results = evaluator.evaluate_one_trajectory(test_poses, estposes, scale=False, kittitype=False)
            seq_name = 'results/' + 'test_' + test_flow[0].split('/')[-1] + '.png'
            plot_traj(results['gt_aligned'], results['est_aligned'], savefigname=seq_name,title='ATE %.4f' % (results['ate_score']))
            # np.savetxt('results/' + 'esti_' + seq_name + '.txt', results['est_aligned'])
            print('\r ATE loss: {}'.format(results['ate_score']))
            print("==> ATE: %.4f" % (results['ate_score']))

if __name__ == '__main__':
    vonet = VONet().cuda()    # test
    data_type = 'tartanair' # euroc or kitti
    model_path = "/home/zhang/tartanvo/models/pose/single_pose_model.train"
    type = "vo"
    VO_test(model=vonet,model_path=model_path, datastr=data_type, imgs_path=test_img, flows_path=test_flow, poses_path=test_pose, test_type=type)

