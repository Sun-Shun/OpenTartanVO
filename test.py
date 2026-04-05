import os
import multiprocessing
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

from Network.VONet import VONet
from Datasets.VODataest import PoseDataset, FlowDataset, VODataset
from Datasets.transformation import motion_ses2pose_quats
from Datasets.utils import (
    Compose, Flow_Crop, CropCenter, DownscaleFlow,
    dataset_intrinsics, load_model, plot_traj,
)
from evaluator.tartanair_evaluator import TartanAirEvaluator


# ─────────────────────────────────────────────
#  Data loading helper  (mirrors train.py)
# ─────────────────────────────────────────────
def collect_test_sequences(root: str):
    """
    Collect sorted (img, flow, pose) sub-directory lists from
    `root/test/`.

    Args:
        root: Dataset root directory, e.g. '/data/tartanair'

    Returns:
        Tuple of three sorted lists: (imgs, flows, poses)
    """
    scenes = sorted(glob(os.path.join(root, 'test')))
    imgs, flows, poses = [], [], []

    for scene in scenes:
        img_dirs  = sorted(glob(os.path.join(scene, 'test_img',  '*')))
        flow_dirs = sorted(glob(os.path.join(scene, 'test_flow', '*')))
        pose_dirs = sorted(glob(os.path.join(scene, 'test_pose', '*')))

        if len(flow_dirs) != len(pose_dirs):
            print(f'[WARNING] flow/pose count mismatch in: {scene}  '
                  f'(flow={len(flow_dirs)}, pose={len(pose_dirs)})')

        imgs  += img_dirs
        flows += flow_dirs
        poses += pose_dirs

    return imgs, flows, poses


# ─────────────────────────────────────────────
#  Main test function
# ─────────────────────────────────────────────
def run_test(
    model: VONet,
    model_path: str,
    datastr: str,
    imgs_path: list,
    flows_path: list,
    poses_path: list,
    test_mode: str,          # renamed from `type` to avoid shadowing builtin
    results_dir: str = 'results',
):
    """
    Evaluate the model on a list of sequences.

    Args:
        model:       Instantiated VONet (already on CUDA).
        model_path:  Path to the checkpoint to load.
        datastr:     Dataset type ('tartanair' / 'euroc' / 'kitti').
        imgs_path:   List of image sub-directories (one per sequence).
        flows_path:  List of flow sub-directories  (one per sequence).
        poses_path:  List of pose sub-directories  (one per sequence).
        test_mode:   One of 'flow', 'pose', 'vo'.
        results_dir: Directory where trajectory plots are saved.
    """
    os.makedirs(results_dir, exist_ok=True)

    pose_std       = np.array([0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=np.float32)
    dict_intrinsic = dataset_intrinsics(datastr)

    # ── Load weights ───────────────────────────
    if test_mode == 'flow':
        load_model(model.flowNet, model_path)
    elif test_mode == 'pose':
        load_model(model.flowPoseNet, model_path)
    elif test_mode == 'vo':
        load_model(model, model_path)
    else:
        raise ValueError(f"test_mode must be 'flow', 'pose', or 'vo', got '{test_mode}'")

    model.eval()

    # ── Iterate over sequences ─────────────────
    for i in range(len(flows_path)):
        seq_tag = os.path.basename(flows_path[i])   # e.g. '01_0000flow'
        print(f'\n[{i + 1}/{len(flows_path)}] Evaluating sequence: {seq_tag}')

        # ── Build dataset for this sequence ────
        if test_mode == 'flow':
            transform = Compose([Flow_Crop((448, 640))])
            dataset   = FlowDataset(
                [imgs_path[i]], [flows_path[i]], None, None,
                transform=transform,
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=5, drop_last=False)

        elif test_mode == 'pose':
            transform = Compose([CropCenter((448, 640), dict_intrinsic), DownscaleFlow()])
            dataset   = PoseDataset(
                None, [flows_path[i]], None, [poses_path[i]],
                transform=transform, is_test=True,
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=5, drop_last=False)

        else:  # vo
            transform = Compose([CropCenter((448, 640), dict_intrinsic), DownscaleFlow()])
            dataset   = VODataset(
                [imgs_path[i]], [flows_path[i]], None, [poses_path[i]],
                transform=transform, is_test=True,
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=5, drop_last=False)

        # ── Inference ──────────────────────────
        if test_mode == 'flow':
            epe_total = 0.0
            with torch.no_grad():
                for batch_idx, data in enumerate(loader, 1):
                    img1 = data['img1'].cuda(non_blocking=True)
                    img2 = data['img2'].cuda(non_blocking=True)
                    flow = data['flow'].cuda(non_blocking=True)

                    pred_flows = model([img1, img2], only_flow=True)
                    pred_flow  = pred_flows[-1].cpu()
                    gt_flow    = flow.detach().cpu()
                    epe_total += torch.mean((pred_flow - gt_flow) ** 2).sqrt().item()

                    print(f'\r  Batch {batch_idx}/{len(loader)}', end='')

            print(f'\n  EPE: {epe_total / len(loader):.4f}')

        else:  # pose or vo
            motion_list = []
            with torch.no_grad():
                for batch_idx, data in enumerate(loader, 1):
                    intrinsic = data['intrinsic'].cuda(non_blocking=True)
                    gt_pose   = data['pose'].cuda(non_blocking=True)

                    if test_mode == 'pose':
                        flow  = data['flow'].cuda(non_blocking=True)
                        pose  = model([flow, intrinsic], only_pose=True)
                    else:  # vo
                        img1  = data['img1'].cuda(non_blocking=True)
                        img2  = data['img2'].cuda(non_blocking=True)
                        flow  = data['flow'].cuda(non_blocking=True)
                        _, pose = model([img1, img2, flow, intrinsic])

                    # Scale estimated translation to match GT magnitude
                    pose_np    = pose.cpu().numpy() * pose_std
                    gt_np      = gt_pose.cpu().numpy()
                    gt_scale   = np.linalg.norm(gt_np[:, :3], axis=1, keepdims=True)
                    est_norm   = np.linalg.norm(pose_np[:, :3], axis=1, keepdims=True)
                    pose_np[:, :3] = pose_np[:, :3] / est_norm * gt_scale
                    motion_list.extend(pose_np.tolist())

                    print(f'\r  Batch {batch_idx}/{len(loader)}', end='')

            # ── Evaluate trajectory ─────────────
            est_poses  = motion_ses2pose_quats(np.array(motion_list))
            evaluator  = TartanAirEvaluator()
            kittitype  = (datastr == 'kitti')
            results    = evaluator.evaluate_one_trajectory(
                poses_path[i], est_poses, scale=False, kittitype=kittitype,
            )
            ate = results['ate_score']

            # ── Save trajectory plot ────────────
            save_path = os.path.join(results_dir, f'test_{seq_tag}.png')
            plot_traj(
                results['gt_aligned'], results['est_aligned'],
                savefigname=save_path,
                title=f'ATE {ate:.4f}',
            )
            print(f'\n  ATE: {ate:.4f}  →  saved to {save_path}')


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':

    # ── Configuration — edit these before running ──
    DATA_ROOT  = '/Data_new/data/tartanair'
    MODEL_PATH = '/tartanvo/models/pose/single_pose_model.train'
    DATA_TYPE  = 'tartanair'   # 'tartanair' | 'euroc' | 'kitti'
    TEST_MODE  = 'vo'          # 'flow'      | 'pose'  | 'vo'
    RESULTS_DIR = 'results'
    # ───────────────────────────────────────────────

    # spawn must be called in __main__, before any CUDA / DataLoader usage
    multiprocessing.set_start_method('spawn', force=True)

    test_imgs, test_flows, test_poses = collect_test_sequences(DATA_ROOT)
    print(f'[Data] Found {len(test_flows)} test sequence(s).')

    vonet = VONet().cuda()

    run_test(
        model       = vonet,
        model_path  = MODEL_PATH,
        datastr     = DATA_TYPE,
        imgs_path   = test_imgs,
        flows_path  = test_flows,
        poses_path  = test_poses,
        test_mode   = TEST_MODE,
        results_dir = RESULTS_DIR,
    )
