# Copyright (c) 2020 Carnegie Mellon University, Wenshan Wang <wenshanw@andrew.cmu.edu>
# For License information please see the LICENSE file in the root directory.
# Credit: Xiangwei Wang https://github.com/TimingSpace

import numpy as np
from scipy.spatial.transform import Rotation as R


def line2mat(line_data):
    mat = np.eye(4)
    mat[0:3, :] = line_data.reshape(3, 4)
    return np.matrix(mat)


def motion2pose(data):
    all_pose = [np.eye(4, 4)]
    pose = np.eye(4, 4)
    for i in range(len(data)):
        pose = pose.dot(data[i])
        all_pose.append(pose)
    return all_pose


def pose2motion(data):
    all_motion = []
    for i in range(len(data) - 1):
        motion = np.linalg.inv(data[i]).dot(data[i + 1])
        all_motion.append(motion)
    return np.array(all_motion)  # N x 4 x 4


def SE2se(SE_data):
    result = np.zeros((6))
    result[0:3] = np.array(SE_data[0:3, 3].T)
    result[3:6] = SO2so(SE_data[0:3, 0:3]).T
    return result


def SO2so(SO_data):
    return R.from_matrix(np.array(SO_data)).as_rotvec()


def so2SO(so_data):
    return R.from_rotvec(so_data).as_matrix()


def se2SE(se_data):
    result_mat = np.matrix(np.eye(4))
    result_mat[0:3, 0:3] = so2SO(se_data[3:6])
    result_mat[0:3, 3] = np.matrix(se_data[0:3]).T
    return result_mat


def se_mean(se_datas):
    all_SE = np.matrix(np.eye(4))
    for i in range(se_datas.shape[0]):
        se = se_datas[i, :]
        SE = se2SE(se)
        all_SE = all_SE * SE
    all_se = SE2se(all_SE)
    mean_se = all_se / se_datas.shape[0]
    return mean_se


def SO2quat(SO_data):
    rr = R.from_matrix(np.array(SO_data))
    return rr.as_quat()


def quat2SO(quat_data):
    return R.from_quat(quat_data).as_matrix()


def pos_quat2SE(quat_data):
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3, 0:3] = np.matrix(SO)
    SE[0:3, 3] = np.matrix(quat_data[0:3]).T
    SE = np.array(SE[0:3, :]).reshape(1, 12)
    return SE


def pos_quats2SEs(quat_datas):
    data_len = quat_datas.shape[0]
    SEs = np.zeros((data_len, 12))
    for i_data in range(data_len):
        SE = pos_quat2SE(quat_datas[i_data, :])
        SEs[i_data, :] = SE
    return SEs


def pos_quats2SE_matrices(quat_datas):
    SEs = []
    for quat in quat_datas:
        SO = R.from_quat(quat[3:7]).as_matrix()
        SE = np.eye(4)
        SE[0:3, 0:3] = SO
        SE[0:3, 3] = quat[0:3]
        SEs.append(SE)
    return SEs


def SE2pos_quat(SE_data):
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_data[0:3, 0:3])
    pos_quat[:3] = SE_data[0:3, 3].T
    return pos_quat
