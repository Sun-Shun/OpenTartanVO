import os
from TartanVO import TartanVO
import argparse
from glob import glob
import multiprocessing

# ###############################相关路径设置######################################
train_scene = glob('/home/zhang/Data_new/data/tartanair/train')
test_scene = glob('/home/zhang/Data_new/data/tartanair/test')
train_scene.sort()
test_scene.sort()
# dataset root directory
train_img, train_flow, train_mask, train_pose = [], [], [], []
test_img, test_flow, test_mask, test_pose = [], [], [], []

for t_scene in train_scene:
    train_img_dir = glob(t_scene + "/train_img" + "/*")
    train_flow_dir = glob(t_scene + "/train_flow" + "/*")
    train_mask_dir = glob(t_scene + "/train_mask" + "/*")
    train_pose_dir = glob(t_scene + "/train_pose" + "/*")
    try:
        assert len(train_flow_dir) == len(train_pose_dir)
    except AssertionError:
        print(t_scene)

    train_img_dir.sort()
    train_flow_dir.sort()
    train_mask_dir.sort()
    train_pose_dir.sort()
    train_img += train_img_dir
    train_flow += train_flow_dir
    train_mask += train_mask_dir
    train_pose += train_pose_dir

for v_scene in test_scene:
    test_img_dir = glob(v_scene + "/test_img" + "/*")
    test_flow_dir = glob(v_scene + "/test_flow" + "/*")
    test_mask_dir = glob(v_scene + "/test_mask" + "/*")
    test_pose_dir = glob(v_scene + "/test_pose" + "/*")
    try:
        assert len(test_flow_dir) == len(test_pose_dir)
    except AssertionError:
        print(v_scene)

    test_img_dir.sort()
    test_flow_dir.sort()
    test_mask_dir.sort()
    test_pose_dir.sort()
    test_img += test_img_dir
    test_flow += test_flow_dir
    test_mask += test_mask_dir
    test_pose += test_pose_dir

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VO')
    parser.add_argument('--train_type', type=str, default='vo', help='Tensorboard_name')
    parser.add_argument('--logs_dir', type=str, default='./runs_test', help='logs_dir')
    parser.add_argument('--root_path', type=str, default='./models', help='save_model_path')
    parser.add_argument('--pose_model', type=str, default='./models/only_pose/single_pose_model.train', help='model_path')
    parser.add_argument('--flow_model', type=str, default='./models/flow/raft-small.pth', help='model_path')
    parser.add_argument('--datastr', default='tartanair', help='dataset_type:euroc or kitti')
    # train
    parser.add_argument('--is_train', type=bool, default=True, help='is_train')
    parser.add_argument('--only_pose', type=bool, default=False, help='only_posenet')
    parser.add_argument('--only_flow', type=bool, default=True, help='only_flow')
    parser.add_argument('--vo', type=bool, default=False, help='vo')
    # optimization
    parser.add_argument('--batch_size', type=int, default=1, help='bath_size')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    args = parser.parse_args()
    vo_model = TartanVO(args, train_img[::200], train_flow[::200], train_mask[::200], train_pose[::200], test_img[::200], test_flow[::200], test_mask[::200], test_pose[::200])
    if args.only_pose:
        multiprocessing.set_start_method('spawn')
        vo_model.pose_train(epochs=100, lr_rate=0.0001, step_size=[30, 50], pre_train=False, weight_decay=1e-4)
    elif args.only_flow:
        vo_model.flow_train(epochs=100, lr_rate=0.0003, step_size=[30, 50], pre_train=True, weight_decay=1e-4)
    else:
        vo_model.vo_train(epochs=100, lr_rate=0.00005, step_size=[10], pre_train=True, weight_decay=1e-4)
