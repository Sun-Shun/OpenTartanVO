#!/usr/bin/python
# Modified by Wenshan Wang
# Software License Agreement (BSD License)
# Copyright (c) 2013, Juergen Sturm, TUM

"""Compute the relative pose error (RPE) between two trajectories."""

import random
import numpy as np


def ominus(a, b):
    """Compute the relative 3D transformation between poses a and b (4x4)."""
    return np.dot(np.linalg.inv(a), b)


def compute_distance(transform):
    """Compute the translational distance of a 4x4 homogeneous matrix."""
    return np.linalg.norm(transform[0:3, 3])


def compute_angle(transform):
    """Compute the rotation angle from a 4x4 homogeneous matrix."""
    return np.arccos(
        min(1, max(-1, (np.trace(transform[0:3, 0:3]) - 1) / 2)))


def distances_along_trajectory(traj):
    """Compute cumulative translational distances along a trajectory."""
    motion = [ominus(traj[i + 1], traj[i]) for i in range(len(traj) - 1)]
    distances = [0]
    total = 0
    for t in motion:
        total += compute_distance(t)
        distances.append(total)
    return distances


def evaluate_trajectory(traj_gt, traj_est, param_max_pairs=10000,
                        param_fixed_delta=False, param_delta=1.00):
    """
    Compute the relative pose error between two trajectories.

    traj_gt / traj_est: lists of 4x4 numpy arrays
    Returns: list of [i, j, trans_error, rot_error]
    """
    if not param_fixed_delta:
        if param_max_pairs == 0 or len(traj_est) < np.sqrt(param_max_pairs):
            pairs = [(i, j) for i in range(len(traj_est))
                     for j in range(len(traj_est))]
        else:
            pairs = [(random.randint(0, len(traj_est) - 1),
                      random.randint(0, len(traj_est) - 1))
                     for _ in range(param_max_pairs)]
    else:
        pairs = []
        for i in range(len(traj_est)):
            j = i + param_delta
            if j < len(traj_est):
                pairs.append((i, j))
        if param_max_pairs != 0 and len(pairs) > param_max_pairs:
            pairs = random.sample(pairs, param_max_pairs)

    result = []
    for i, j in pairs:
        error44 = ominus(
            ominus(traj_est[j], traj_est[i]),
            ominus(traj_gt[j],  traj_gt[i]))

        trans = compute_distance(error44)
        rot   = compute_angle(error44)
        result.append([i, j, trans, rot])

    if len(result) < 2:
        raise Exception("Could not find enough pairs between ground truth and estimated trajectory!")

    return result
