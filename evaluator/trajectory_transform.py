# Copyright (c) 2020 Carnegie Mellon University, Wenshan Wang <wenshanw@andrew.cmu.edu>
# For License information please see the LICENSE file in the root directory.

import numpy as np
from .transformation import pos_quats2SE_matrices, SE2pos_quat, pose2motion, motion2pose


def shift0(traj):
    """Translate and rotate trajectory so it starts at the origin."""
    traj_ses = pos_quats2SE_matrices(np.array(traj))
    traj_init_inv = np.linalg.inv(traj_ses[0])
    new_traj = []
    for tt in traj_ses:
        ttt = traj_init_inv.dot(tt)
        new_traj.append(SE2pos_quat(ttt))
    return np.array(new_traj)


def ned2cam(traj):
    """Convert a NED-frame trajectory to camera-frame trajectory."""
    T = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    T_inv = np.linalg.inv(T)
    new_traj = []
    traj_ses = pos_quats2SE_matrices(np.array(traj))
    for tt in traj_ses:
        ttt = T.dot(tt).dot(T_inv)
        new_traj.append(SE2pos_quat(ttt))
    return np.array(new_traj)


def cam2ned(traj):
    """Convert a camera-frame trajectory to NED-frame trajectory."""
    T = np.array([[0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    T_inv = np.linalg.inv(T)
    new_traj = []
    traj_ses = pos_quats2SE_matrices(np.array(traj))
    for tt in traj_ses:
        ttt = T.dot(tt).dot(T_inv)
        new_traj.append(SE2pos_quat(ttt))
    return np.array(new_traj)


def trajectory_transform(gt_traj, est_traj):
    """Centre both trajectories at the origin."""
    gt_traj_trans = shift0(gt_traj)
    est_traj_trans = shift0(est_traj)
    return gt_traj_trans, est_traj_trans


def pose2trans(pose_data):
    trans = []
    for i in range(len(pose_data) - 1):
        tran = np.array(pose_data[i + 1][:3]) - np.array(pose_data[i][:3])
        trans.append(tran)
    return np.array(trans)  # N x 3


def rescale(poses_gt, poses):
    """Compute and apply a global scale factor to align *poses* with *poses_gt*.

    poses_gt/poses: N x 7 pose list in quaternion format [x y z qx qy qz qw]
    Returns: (scaled poses, scale factor)
    """
    trans_gt = pose2trans(poses_gt)
    trans = pose2trans(poses)

    speed_gt = np.sqrt(np.sum(trans_gt * trans_gt, 1))
    speed = np.sqrt(np.sum(trans * trans, 1))

    mask = speed_gt > 0.0001
    scale = np.mean(speed[mask] / speed_gt[mask])
    scale = 1.0 / scale
    poses[:, 0:3] = poses[:, 0:3] * scale
    return poses, scale


def trajectory_scale(traj, scale):
    for ttt in traj:
        ttt[0:3, 3] = ttt[0:3, 3] * scale
    return traj
