import torch
from torch.utils.data import Dataset
import numpy as np
from os import listdir
import cv2
from Datasets.utils import make_intrinsics_layer
from evaluator.transformation import pos_quats2SEs, pose2motion, SEs2ses


class BaseDataset(Dataset):
    def __init__(self, img_dir_list, flow_dir_list, mask_dir_list, pose_file_list, is_test=False):

        self.img_path_list = img_dir_list
        self.flow_path_list = flow_dir_list
        self.mask_path_list = mask_dir_list
        self.pose_list = pose_file_list
        self.is_test = is_test
        self.datas = self.merge_data(self.img_path_list, self.flow_path_list, self.mask_path_list, self.pose_list,self.is_test)

    def get_img_data(self, dir_path):
        path_list = []
        img_paths = listdir(dir_path)
        img_paths_list = [(dir_path + '/' + ff) for ff in img_paths if (ff.endswith('.png') or ff.endswith('.jpg'))]
        img_paths_list.sort()
        path_list += img_paths_list
        return path_list

    def get_flow_data(self, dir_path):

        path_list = []
        flow_paths = listdir(dir_path)
        flow_paths_list = [(dir_path + '/' + ff) for ff in flow_paths if ff.endswith('.npy')]
        flow_paths_list.sort()
        path_list += flow_paths_list
        return path_list

    def get_mask_data(self, dir_path):

        path_list = []
        mask_paths = listdir(dir_path)
        mask_paths_list = [(dir_path + '/' + ff) for ff in mask_paths if ff.endswith('.npy')]
        mask_paths_list.sort()
        path_list += mask_paths_list
        return path_list

    def get_pose_data(self, files_path, is_test=False):

        all_pose_list = []
        pose_std = [0.13, 0.13, 0.13, 0.013, 0.013, 0.013]
        poses_list = []
        motions = []
        if files_path is not None:
            poselist = np.loadtxt(files_path).astype(np.float32)
            assert (poselist.shape[1] == 7)  # position + quaternion
            poses = pos_quats2SEs(poselist)
            matrix = pose2motion(poses)
            motions = SEs2ses(matrix).astype(np.float32)
            if not is_test:
                motions = motions / pose_std
                trans = motions[:, :3]
                trans_norm = np.linalg.norm(trans, axis=1)
                data_fine = trans_norm.reshape(-1, 1) + float(1e-15)
                motions[:, :3] = motions[:, :3] / data_fine
                print("motion norm /pose_std")
        for j in range(int(len(motions))):
            poses_motion = motions[j]
            poses_list.append(poses_motion)
        all_pose_list += poses_list
        return all_pose_list

    def merge_data(self, img_path_list, flow_path_list, mask_path_list, pose_data_list, is_test=False):
        sampler_datas = []
        # only train PoseNet
        if img_path_list is None:
            if mask_path_list is None:
                for i in range(0, len(flow_path_list)):
                    flow_path = self.get_flow_data(flow_path_list[i])
                    pose_path = self.get_pose_data(pose_data_list[i],is_test)
                    for j in range(0, len(flow_path)):
                        sampler_datas.append((flow_path[j], pose_path[j]))

            else:
                for i in range(0, len(flow_path_list)):
                    flow_path = self.get_flow_data(flow_path_list[i])
                    mask_path = self.get_mask_data(mask_path_list[i])
                    pose_path = self.get_pose_data(pose_data_list[i],is_test)
                    for j in range(0, len(flow_path)):
                        sampler_datas.append((flow_path[j], mask_path[j], pose_path[j+1]))
        # only train FlowNet
        elif pose_data_list is None:
            if mask_path_list is None:
                for i in range(0, len(flow_path_list)):
                    imgs_path = self.get_img_data(img_path_list[i])
                    flow_path = self.get_flow_data(flow_path_list[i])
                    for j in range(0, len(flow_path)):
                        sampler_datas.append((imgs_path[j], imgs_path[j+1], flow_path[j]))
            else:
                for i in range(0, len(flow_path_list)):
                    imgs_path = self.get_img_data(img_path_list[i])
                    flow_path = self.get_flow_data(flow_path_list[i])
                    mask_path = self.get_mask_data(mask_path_list[i])
                    for j in range(0, len(flow_path)):
                        sampler_datas.append((imgs_path[j], imgs_path[j+1], flow_path[j], mask_path[j]))
        # train VO
        else:
            if mask_path_list is None:

                for i in range(0, len(flow_path_list)):
                    imgs_path = self.get_img_data(img_path_list[i])
                    flow_path = self.get_flow_data(flow_path_list[i])
                    pose_path = self.get_pose_data(pose_data_list[i],is_test)
                    for j in range(0, len(flow_path)):
                        sampler_datas.append((imgs_path[j], imgs_path[j+1], flow_path[j], pose_path[j]))

            else:
                for i in range(0, len(flow_path_list)):
                    imgs_path = self.get_img_data(img_path_list[i])
                    flow_path = self.get_flow_data(flow_path_list[i])
                    mask_path = self.get_mask_data(mask_path_list[i])
                    pose_path = self.get_pose_data(pose_data_list[i],is_test)
                    for j in range(0, len(flow_path)):
                        sampler_datas.append((imgs_path[j], imgs_path[j+1], flow_path[j], mask_path[j], pose_path[j]))

        return sampler_datas

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class PoseDataset(BaseDataset):
    def __init__(self, img_dir_list, flow_dir_list, mask_dir_list, pose_file_list, transform=None, is_test=False):
        super(PoseDataset, self).__init__(img_dir_list, flow_dir_list, mask_dir_list, pose_file_list, is_test)
        self.transform = transform

    def __getitem__(self, index):
        '''
            data augmentation to gpu
            datas[0] is gt_flow
            datas[1] is flow_mask
            datas[2] is pose
            gt_flow: tensor [C,H,W]
        '''

        # no mask be used
        if self.mask_path_list is None:
            flow_path = self.datas[index][0]
            flow = torch.as_tensor(np.load(flow_path)).permute((2, 0, 1)).cuda()
            pose = torch.as_tensor(self.datas[index][1]).cuda()
            res = {'flow': flow, 'pose': pose}
        else:
            flow_path = self.datas[index][0]
            mask_path = self.datas[index][1]
            flow = torch.as_tensor(np.load(flow_path)).permute((2, 0, 1)).cuda()
            mask = torch.as_tensor(np.load(mask_path)).unsqueeze(2).permute((2, 0, 1)).cuda()
            pose = torch.as_tensor(self.datas[index][1]).cuda()
            res = {'flow': flow, 'mask': mask, 'pose': pose}

        if self.transform:
            res = self.transform(res)

        return res

    def __len__(self):
        return len(self.datas)


class FlowDataset(BaseDataset):
    def __init__(self, img_dir_list, flow_dir_list, mask_dir_list, pose_file_list, transform=None):
        super(FlowDataset, self).__init__(img_dir_list, flow_dir_list, mask_dir_list, pose_file_list)
        self.transform = transform
        try:
            len(self.flow_path_list) + len(flow_dir_list) == len(self.img_path_list)
        except AssertionError:
            print('flow and img not matching!')

    def __getitem__(self, index):
        '''
            datas[0] is img1
            datas[1] is img2
            datas[2] is gt_flow
            datas[3] is mask
        '''
        if self.mask_path_list is None:
            img1_path, img2_path = self.datas[index][0], self.datas[index][1]
            flow_path = self.datas[index][2]
            flow = torch.as_tensor(np.load(flow_path)).permute((2, 0, 1))
            img_1 = torch.as_tensor(cv2.imread(img1_path)).permute((2, 0, 1)).float()
            img_2 = torch.as_tensor(cv2.imread(img2_path)).permute((2, 0, 1)).float()
            res = {'img1': img_1, 'img2': img_2, 'flow': flow}
        else:
            img1_path, img2_path = self.datas[index][0], self.datas[index][1]
            flow_path, mask_path = self.datas[index][2], self.datas[index][3]
            flow = torch.as_tensor(np.load(flow_path)).permute((2, 0, 1))
            mask = torch.as_tensor(np.load(mask_path)).unsqueeze(2).permute((2, 0, 1))
            img_1 = torch.as_tensor(cv2.imread(img1_path)).permute((2, 0, 1)).float()
            img_2 = torch.as_tensor(cv2.imread(img2_path)).permute((2, 0, 1)).float()
            res = {'img1': img_1, 'img2': img_2, 'flow': flow, 'mask': mask.float()}
        if self.transform:
            res = self.transform(res)

        return res

    def __len__(self):
        return len(self.datas)


class VODataset(BaseDataset):
    def __init__(self, img_dir_list, flow_dir_list, mask_dir_list, pose_file_list, transform=None, is_test=False):
        super(VODataset, self).__init__(img_dir_list, flow_dir_list, mask_dir_list, pose_file_list, is_test)
        self.transform = transform

    def __getitem__(self, index):
        '''
            datas[0] is img1:[C,H,W]
            datas[1] is img2:[C,H,W]
            datas[2] is gt_flow:[C,H,W]
            datas[3] is mask:[C,H,W]
            datas[4] is pose
        '''

        if self.mask_path_list is None:
            img1_path, img2_path = self.datas[index][0], self.datas[index][1]
            flow_path = self.datas[index][2]
            pose = self.datas[index][3]
            pose = torch.as_tensor(pose)
            flow = torch.as_tensor(np.load(flow_path)).permute((2, 0, 1))
            img_1 = torch.as_tensor(cv2.imread(img1_path)).permute((2, 0, 1)).float()
            img_2 = torch.as_tensor(cv2.imread(img2_path)).permute((2, 0, 1)).float()
            res = {'img1': img_1, 'img2': img_2, 'flow': flow, 'pose': pose}
        else:
            img1_path, img2_path = self.datas[index][0], self.datas[index][1]
            flow_path, mask_path = self.datas[index][2], self.datas[index][3]
            pose = self.datas[index][4]
            pose = torch.as_tensor(pose)
            flow = torch.as_tensor(np.load(flow_path)).permute((2, 0, 1))
            mask = torch.as_tensor(np.load(mask_path)).unsqueeze(2).permute((2, 0, 1))
            img_1 = torch.as_tensor(cv2.imread(img1_path)).permute((2, 0, 1)).float()
            img_2 = torch.as_tensor(cv2.imread(img2_path)).permute((2, 0, 1)).float()
            res = {'img1': img_1, 'img2': img_2, 'flow': flow, 'pose': pose, 'mask': mask.float()}

        if self.transform:
            res = self.transform(res)

        return res

    def __len__(self):
        return len(self.datas)
