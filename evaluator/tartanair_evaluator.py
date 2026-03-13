# Copyright (c) 2020 Carnegie Mellon University, Wenshan Wang <wenshanw@andrew.cmu.edu>
# For License information please see the LICENSE file in the root directory.

import numpy as np
from .evaluator_base import ATEEvaluator, RPEEvaluator, KittiEvaluator, transform_trajs, quats2SEs


class TartanAirEvaluator:
    def __init__(self):
        self.ate_eval   = ATEEvaluator()
        self.rpe_eval   = RPEEvaluator()
        self.kitti_eval = KittiEvaluator()

    def evaluate_one_trajectory(self, gt_traj, est_traj, scale=False, kittitype=True):
        """
        Evaluate a single estimated trajectory against ground truth.

        gt_traj  -- ground-truth poses (N x 7 array or path to .txt)
        est_traj -- estimated poses    (N x 7 array or path to .txt)
        scale    -- True: compute and apply a global scale factor (monocular VO)
        kittitype-- True: use KITTI segment lengths (100-800 m),
                    False: use shorter lengths (5-40 m)

        Returns dict with keys:
            ate_score, rpe_score, kitti_score, gt_aligned, est_aligned
        """
        try:
            gt_traj  = np.loadtxt(gt_traj)
            est_traj = np.loadtxt(est_traj)
        except TypeError:
            pass  # already numpy arrays

        if gt_traj.shape[0] != est_traj.shape[0]:
            raise Exception("POSEFILE_LENGTH_ILLEGAL")
        if gt_traj.shape[1] != 7 or est_traj.shape[1] != 7:
            raise Exception("POSEFILE_FORMAT_ILLEGAL")

        gt_traj_trans, est_traj_trans, s = transform_trajs(gt_traj, est_traj, scale)
        gt_SEs, est_SEs = quats2SEs(gt_traj_trans, est_traj_trans)

        ate_score, gt_ate_aligned, est_ate_aligned = self.ate_eval.evaluate(
            gt_traj, est_traj, scale)
        rpe_score   = self.rpe_eval.evaluate(gt_SEs, est_SEs)
        kitti_score = self.kitti_eval.evaluate(gt_SEs, est_SEs, kittitype=kittitype)

        return {
            'ate_score':   ate_score,
            'rpe_score':   rpe_score,
            'kitti_score': kitti_score,
            'gt_aligned':  gt_ate_aligned,
            'est_aligned': est_ate_aligned,
        }


if __name__ == "__main__":
    evaluator = TartanAirEvaluator()
    result = evaluator.evaluate_one_trajectory('pose_gt.txt', 'pose_est.txt', scale=True)
    print(result)
