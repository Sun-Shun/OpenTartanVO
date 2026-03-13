# Copyright (c) 2020 Carnegie Mellon University, Wenshan Wang <wenshanw@andrew.cmu.edu>
# For License information please see the LICENSE file in the root directory.
# Python re-implementation of the KITTI odometry metric.
# Credit: Xiangwei Wang https://github.com/TimingSpace

import numpy as np


def trajectory_distances(poses):
    distances = [0]
    for i in range(1, len(poses)):
        delta = poses[i - 1][0:3, 3] - poses[i][0:3, 3]
        distances.append(distances[i - 1] + np.linalg.norm(delta))
    return distances


def last_frame_from_segment_length(dist, first_frame, length):
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + length:
            return i
    return -1


def rotation_error(pose_error):
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1)
    return np.arccos(max(min(d, 1.0), -1.0))


def translation_error(pose_error):
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_sequence_error(poses_gt, poses_result,
                              lengths=None):
    if lengths is None:
        lengths = [10, 20, 30, 40, 50, 60, 70, 80]

    errors = []
    step_size = 1
    dist = trajectory_distances(poses_gt)

    for first_frame in range(0, len(poses_gt), step_size):
        for length in lengths:
            last_frame = last_frame_from_segment_length(dist, first_frame, length)
            if last_frame == -1:
                continue

            pose_delta_gt     = np.linalg.inv(poses_gt[first_frame]).dot(poses_gt[last_frame])
            pose_delta_result = np.linalg.inv(poses_result[first_frame]).dot(poses_result[last_frame])
            pose_error        = np.linalg.inv(pose_delta_result).dot(pose_delta_gt)

            r_err = rotation_error(pose_error)
            t_err = translation_error(pose_error)
            num_frames = float(last_frame - first_frame + 1)
            speed = length / (0.1 * num_frames)
            errors.append([first_frame, r_err / length, t_err / length, length, speed])

    return errors


def calculate_ave_errors(errors, lengths=None):
    if lengths is None:
        lengths = [10, 20, 30, 40, 50, 60, 70, 80]

    rot_errors = []
    tra_errors = []
    for length in lengths:
        rot_each = [e[1] for e in errors if abs(e[3] - length) < 0.1]
        tra_each = [e[2] for e in errors if abs(e[3] - length) < 0.1]
        if not rot_each:
            continue
        rot_errors.append(sum(rot_each) / len(rot_each))
        tra_errors.append(sum(tra_each) / len(tra_each))

    return np.array(rot_errors) * 180 / np.pi, tra_errors


def evaluate(gt, data, kittitype=True):
    if kittitype:
        lens = [100, 200, 300, 400, 500, 600, 700, 800]
    else:
        lens = [5, 10, 15, 20, 25, 30, 35, 40]
    errors = calculate_sequence_error(gt, data, lengths=lens)
    rot, tra = calculate_ave_errors(errors, lengths=lens)
    return np.mean(rot), np.mean(tra)
