import os
import argparse
import multiprocessing
from glob import glob

from TartanVO import TartanVO


# ─────────────────────────────────────────────
#  Helper: argparse-compatible boolean parsing
# ─────────────────────────────────────────────
def str2bool(v: str) -> bool:
    """
    Fix argparse bool bug: bool("False") == True.
    Now --flag False / --flag true / --flag 0 all work as expected.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


# ─────────────────────────────────────────────
#  Data loading helper
# ─────────────────────────────────────────────
def collect_sequences(root: str, split: str):
    """
    Walk through all scene folders under `root/<split>/` and collect
    sorted lists of (img, flow, mask, pose) sub-directories.

    Args:
        root:  Dataset root directory, e.g. '/data/tartanair'
        split: 'train' or 'test'

    Returns:
        Tuple of four sorted lists: (imgs, flows, masks, poses)
    """
    scenes = sorted(glob(os.path.join(root, split)))
    imgs, flows, masks, poses = [], [], [], []

    for scene in scenes:
        img_dirs  = sorted(glob(os.path.join(scene, f'{split}_img',  '*')))
        flow_dirs = sorted(glob(os.path.join(scene, f'{split}_flow', '*')))
        mask_dirs = sorted(glob(os.path.join(scene, f'{split}_mask', '*')))  # optional
        pose_dirs = sorted(glob(os.path.join(scene, f'{split}_pose', '*')))

        if len(flow_dirs) != len(pose_dirs):
            print(f'[WARNING] flow/pose count mismatch in: {scene}  '
                  f'(flow={len(flow_dirs)}, pose={len(pose_dirs)})')

        imgs  += img_dirs
        flows += flow_dirs
        masks += mask_dirs
        poses += pose_dirs

    return imgs, flows, masks, poses


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='OpenTrain — TartanVO Training')

    # ── Paths ──────────────────────────────────
    parser.add_argument('--data_root',   type=str,
                        default='/home/zhang/Data_new/data/tartanair',
                        help='Dataset root directory (contains train/ and test/)')
    parser.add_argument('--logs_dir',    type=str, default='./runs_test',
                        help='TensorBoard log directory')
    parser.add_argument('--root_path',   type=str, default='./models',
                        help='Directory to save checkpoints')
    parser.add_argument('--pose_model',  type=str,
                        default='./models/only_pose/single_pose_model.train',
                        help='Pre-trained pose model path')
    parser.add_argument('--flow_model',  type=str,
                        default='./models/flow/raft-small.pth',
                        help='Pre-trained flow model path')

    # ── Dataset ────────────────────────────────
    parser.add_argument('--datastr',     type=str, default='tartanair',
                        choices=['tartanair', 'euroc', 'kitti'],
                        help='Dataset type')
    parser.add_argument('--sample_step', type=int, default=200,
                        help='Sub-sample step for quick experiments (1 = use all data)')

    # ── Training mode ──────────────────────────
    parser.add_argument('--is_train',   type=str2bool, default=True)
    parser.add_argument('--only_flow',  type=str2bool, default=False,
                        help='Train flow network only')
    parser.add_argument('--only_pose',  type=str2bool, default=False,
                        help='Train pose network only')
    parser.add_argument('--vo',         type=str2bool, default=True,
                        help='Train full end-to-end VO network')

    # ── Optimisation ───────────────────────────
    parser.add_argument('--batch_size',  type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()

    # ── Validate training mode flags ───────────
    active = sum([args.only_flow, args.only_pose, args.vo])
    if active != 1:
        parser.error('Exactly one of --only_flow / --only_pose / --vo must be True.')

    # ── Collect data ───────────────────────────
    step = args.sample_step
    train_imgs, train_flows, train_masks, train_poses = collect_sequences(args.data_root, 'train')
    test_imgs,  test_flows,  test_masks,  test_poses  = collect_sequences(args.data_root, 'test')

    train_imgs,  train_flows,  train_masks,  train_poses  = (
        train_imgs[::step], train_flows[::step], train_masks[::step], train_poses[::step]
    )
    test_imgs,   test_flows,   test_masks,   test_poses   = (
        test_imgs[::step],  test_flows[::step],  test_masks[::step],  test_poses[::step]
    )

    print(f'[Data] train sequences: {len(train_flows)}  |  test sequences: {len(test_flows)}')

    # ── Build model ────────────────────────────
    vo_model = TartanVO(
        args,
        train_imgs,  train_flows,  train_masks,  train_poses,
        test_imgs,   test_flows,   test_masks,   test_poses,
    )

    # ── Launch training ────────────────────────
    if args.only_pose:
        # spawn is required for CUDA + DataLoader with multiple workers
        multiprocessing.set_start_method('spawn')
        vo_model.pose_train(epochs=100, lr_rate=1e-4, step_size=[30, 50],
                            pre_train=False, weight_decay=1e-4)

    elif args.only_flow:
        vo_model.flow_train(epochs=100, lr_rate=3e-4, step_size=[30, 50],
                            pre_train=True, weight_decay=1e-4)

    else:  # full VO
        vo_model.vo_train(epochs=100, lr_rate=5e-5, step_size=[10],
                          pre_train=True, weight_decay=1e-4)
