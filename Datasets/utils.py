from __future__ import division
import torch
import math
import random
import numpy as np
import numbers
import cv2
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F


# ===== general functions =====

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img







class Flow_RRCrop(object):
    """
    Random scale to cover continuous focal length
    Due to the tartanair focal is already small, we only up scale the image

    """
    '''
    transformlist = [RandomResizeCrop(size=(image_height, image_width), max_scale=max_intrinsic / 320.0,
                                      keep_center=args.random_crop_center, fix_ratio=args.fix_ratio)]
    max_intrinsic = 0
    args.random_crop_center = True
    rgs.fix_ratio = True
    '''

    def __init__(self, size, max_scale=2.5,keep_center=False, fix_ratio=False):
        '''
        size: output frame size, this should be NO LARGER than the input frame size!
        scale_disp: when training the stereovo, disparity represents depth, which is not scaled with resize
        '''
        if isinstance(size, numbers.Number):
            self.target_h = int(size)
            self.target_w = int(size)
        else:
            self.target_h = size[0]
            self.target_w = size[1]

        self.keep_center = keep_center
        self.fix_ratio = fix_ratio
        self.scale_base = max_scale  # self.max_focal /self.tartan_focal

    def __call__(self, sample):
        for kk in sample:
            if len(sample[kk].shape) >= 2:
                h, w = sample[kk].shape[1], sample[kk].shape[2]
                break

        scale_w, scale_h, x1, y1, crop_w, crop_h = generate_random_scale_crop(h, w, self.scale_base,
                                                                              self.keep_center,self.fix_ratio)

        scale_h_2, scale_w_2, scale_2 = 1., 1., 1.
        x1_2 = int((crop_w - self.target_w) / 2)
        y1_2 = int((crop_h - self.target_h) / 2)
        if self.target_h > crop_h:
            scale_h_2 = float(self.target_h) / crop_h
        if self.target_w > crop_w:
            scale_w_2 = float(self.target_w) / crop_w
        if scale_h_2 > 1 or scale_w_2 > 1:
            scale_2 = max(scale_h_2, scale_w_2)
            crop_w_2 = int(round(crop_w * scale_2))
            crop_h_2 = int(round(crop_h * scale_2))
            x1_2 = int((crop_w_2 - self.target_w) / 2)
            y1_2 = int((crop_h_2 - self.target_h) / 2)

        for nn in sample:
            if nn not in ['pose'] :
                # 1st scale_crop
                # scale flow
                sample[nn] = F.interpolate(sample[nn].unsqueeze(0), scale_factor=(scale_h, scale_w), mode='bilinear')
                sample[nn] = sample[nn].squeeze(0)
                # crop flow
                sample[nn] = sample[nn][:, y1:y1 + crop_h, x1:x1 + crop_w]
                # 2st crop_hw<target_hw
                if scale_2 > 1.0:
                    sample[nn] = F.interpolate(sample[nn].unsqueeze(0), size=(crop_h_2, crop_w_2), mode='bilinear')
                    sample[nn] = sample[nn].squeeze(0)
                # align target_hw
                sample[nn] = sample[nn][:, y1_2:y1_2 + int(self.target_h), x1_2:x1_2 + int(self.target_w)]

        # scale optical flow
        if 'flow' in sample:
            sample['flow'][0, :, :] = sample['flow'][0, :, :] * scale_w * scale_2
            sample['flow'][1, :, :] = sample['flow'][1, :, :] * scale_h * scale_2

        return sample

class Flow_Crop(object):
    """Crops the a sample of data (tuple) at center
    if the image size is not large enough, it will be first resized with fixed ratio
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size


    def __call__(self, sample):

        kks = list(sample.keys())

        th, tw = self.size
        h, w = sample[kks[0]].shape[1], sample[kks[0]].shape[2]

        scale_h, scale_w, scale = 1., 1., 1.
        if th > h:
            scale_h = float(th) / h
        if tw > w:
            scale_w = float(tw) / w
        if scale_h > 1 or scale_w > 1:
            scale = max(scale_h, scale_w)
            w = int(round(w * scale))  # w after resize
            h = int(round(h * scale))  # h after resize

        x1 = int((w - tw) / 2)
        y1 = int((h - th) / 2)

        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape) == 3:
                if scale > 1:
                    img = F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
                    img = img.squeeze(0)
                sample[kk] = img[:, y1:y1 + th, x1:x1 + tw]

            elif len(img.shape) == 2:
                if scale > 1:
                    img = img.squeeze(0)
                    img = F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
                sample[kk] = img[:, y1:y1 + th, x1:x1 + tw]

        if "flow" in sample:
            sample["flow"][0, :, :] = sample["flow"][0, :, :] * scale
            sample["flow"][1, :, :] = sample["flow"][1, :, :] * scale

        return sample

def generate_random_scale_crop(h, w, scale_base, keep_center, fix_ratio):
    """
    Randomly generate scale and crop params
    H: input image h
    w: input image w
    target_h: output image h
    target_w: output image w
    scale_base: max scale up rate
    keep_center: crop at center
    fix_ratio: scale_h == scale_w
    """

    # Generate random scale
    scale_s_w = random.random() * (scale_base - 1) + 1

    if fix_ratio:
        scale_s_h = scale_s_w
    else:
        scale_s_h = random.random() * (scale_base - 1) + 1

    n_w = scale_s_w * w
    n_h = scale_s_h * h

    # Generate random crop
    scale_c_w = 0.5*random.random()+1.0
    scale_c_h = 0.5*random.random()+1.0
    crop_w = int(math.ceil(n_w / scale_c_w))  #
    crop_h = int(math.ceil(n_h / scale_c_h))  #

    if keep_center:  # True
        x1 = int((n_w - crop_w) / 2)
        y1 = int((n_h - crop_h) / 2)
    else:
        x1 = random.randint(0, int(n_w - crop_w))
        y1 = random.randint(0, int(n_h - crop_h))

    return scale_s_w, scale_s_h, x1, y1, crop_w, crop_h


class RandomResizeCrop(object):
    """
    Random scale to cover continuous focal length
    Due to the tartanair focal is already small, we only up scale the image

    """
    '''
    transformlist = [RandomResizeCrop(size=(image_height, image_width), max_scale=max_intrinsic / 320.0,
                                      keep_center=args.random_crop_center, fix_ratio=args.fix_ratio)]
    max_intrinsic = 0
    args.random_crop_center = True
    rgs.fix_ratio = True
    '''

    def __init__(self, size, intrinsic, max_scale=2.5,
                 keep_center=False, fix_ratio=False,use_gpu=False):
        '''
        size: output frame size, this should be NO LARGER than the input frame size!
        scale_disp: when training the stereovo, disparity represents depth, which is not scaled with resize
        '''
        if isinstance(size, numbers.Number):
            self.target_h = int(size)
            self.target_w = int(size)
        else:
            self.target_h = size[0]
            self.target_w = size[1]

        self.keep_center = keep_center
        self.fix_ratio = fix_ratio
        self.fx = intrinsic[0]
        self.fy = intrinsic[1]
        self.cx = intrinsic[2]
        self.cy = intrinsic[3]
        self.use_gpu = use_gpu
        self.scale_base = max_scale  # self.max_focal /self.tartan_focal

    def __call__(self, sample):
        for kk in sample:
            if len(sample[kk].shape) >= 2:
                h, w = sample[kk].shape[1], sample[kk].shape[2]
                break

        scale_w, scale_h, x1, y1, crop_w, crop_h = generate_random_scale_crop(h, w, self.scale_base,
                                                                              self.keep_center,self.fix_ratio)

        scale_h_2, scale_w_2, scale_2 = 1., 1., 1.
        x1_2 = int((crop_w - self.target_w) / 2)
        y1_2 = int((crop_h - self.target_h) / 2)
        if self.target_h > crop_h:
            scale_h_2 = float(self.target_h) / crop_h
        if self.target_w > crop_w:
            scale_w_2 = float(self.target_w) / crop_w
        if scale_h_2 > 1 or scale_w_2 > 1:
            scale_2 = max(scale_h_2, scale_w_2)
            crop_w_2 = int(round(crop_w * scale_2))
            crop_h_2 = int(round(crop_h * scale_2))
            x1_2 = int((crop_w_2 - self.target_w) / 2)
            y1_2 = int((crop_h_2 - self.target_h) / 2)

        for nn in sample:
            if nn not in ['pose']:
                # 1st scale_crop
                # scale flow
                sample[nn] = F.interpolate(sample[nn].unsqueeze(0), scale_factor=(scale_h, scale_w), mode='bilinear')
                sample[nn] = sample[nn].squeeze(0)
                # crop flow
                sample[nn] = sample[nn][:, y1:y1 + crop_h, x1:x1 + crop_w]
                # 2st crop_hw<target_hw
                if scale_2 > 1.0:
                    sample[nn] = F.interpolate(sample[nn].unsqueeze(0), size=(crop_h_2, crop_w_2), mode='bilinear')
                    sample[nn] = sample[nn].squeeze(0)
                # align target_hw
                sample[nn] = sample[nn][:, y1_2:y1_2 + int(self.target_h), x1_2:x1_2 + int(self.target_w)]

        # Generate intrinsics
        sample['intrinsic'] = make_intrinsics_layer(self.target_w, self.target_h, self.fx * scale_w * scale_2,
                                                    self.fy * scale_h * scale_2, self.target_w / 2.0, self.target_h / 2.0, use_gpu=self.use_gpu)
        # scale optical flow
        if 'flow' in sample:
            sample['flow'][0, :, :] = sample['flow'][0, :, :] * scale_w * scale_2
            sample['flow'][1, :, :] = sample['flow'][1, :, :] * scale_h * scale_2

        return sample


class CropCenter(object):
    """Crops the a sample of data (tuple) at center
    if the image size is not large enough, it will be first resized with fixed ratio
    """

    def __init__(self, size, intrinsic):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fx = intrinsic[0]
        self.fy = intrinsic[1]
        self.cx = intrinsic[2]
        self.cy = intrinsic[3]

    def __call__(self, sample):

        kks = list(sample.keys())

        th, tw = self.size
        h, w = sample[kks[0]].shape[1], sample[kks[0]].shape[2]

        scale_h, scale_w, scale = 1., 1., 1.
        if th > h:
            scale_h = float(th) / h
        if tw > w:
            scale_w = float(tw) / w
        if scale_h > 1 or scale_w > 1:
            scale = max(scale_h, scale_w)
            w = int(round(w * scale))  # w after resize
            h = int(round(h * scale))  # h after resize

        x1 = int((w - tw) / 2)
        y1 = int((h - th) / 2)

        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape) == 3:
                if scale > 1:
                    img = F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
                    img = img.squeeze(0)
                sample[kk] = img[:, y1:y1 + th, x1:x1 + tw]

        sample['intrinsic'] = make_intrinsics_layer(tw, th, self.fx * scale, self.fy * scale, tw / 2.0, th / 2.0)

        if "flow" in sample:
            sample["flow"][0, :, :] = sample["flow"][0, :, :] * scale
            sample["flow"][1, :, :] = sample["flow"][1, :, :] * scale

        return sample

class DownscaleFlow(object):
    """
    Scale the flow and mask to a fixed size

    """

    def __init__(self, scale=4):
        '''
        size: output frame size, this should be NO LARGER than the input frame size!
        '''
        self.downscale = 1.0 / scale

    def __call__(self, sample):
        if self.downscale != 1 and 'flow' in sample:
            sample['flow'] = F.interpolate(sample['flow'].unsqueeze(0), scale_factor=self.downscale, mode='bilinear',
                                           align_corners=False)
            sample['flow'] = sample['flow'].squeeze(0)

        if self.downscale != 1 and 'intrinsic' in sample:
            sample['intrinsic'] = F.interpolate(sample['intrinsic'].unsqueeze(0), scale_factor=self.downscale,
                                                mode='bilinear', align_corners=False)
            sample['intrinsic'] = sample['intrinsic'].squeeze(0)

        return sample


class ToTensor(object):
    def __call__(self, sample):

        kks = list(sample)

        for kk in kks:
            data = sample[kk]

            # if len(data.shape) == 3:  # transpose image-like data
            #     data = data.permute(2, 0, 1)
            if len(data.shape) == 2:
                data = data.reshape((1,) + data.shape)

            elif len(data.shape) == 3 and data.shape[0] == 3:  # normalization of rgb images
                data = data / 255.0

            else:
                pass

            sample[kk] = data

        return sample


def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2(dv, du)

    angleShift = np.pi

    if (True == flagDegree):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt(du * du + dv * dv)

    return a, d, angleShift


def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0):
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = calculate_angle_distance_from_du_dv(flownp[:, :, 0], flownp[:, :, 1], flagDegree=False)

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros((ang.shape[0], ang.shape[1], 3), dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[:, :, 0] = np.remainder((ang + angShift) / (2 * np.pi), 1)
    hsv[:, :, 1] = mag / maxF * n
    hsv[:, :, 2] = (n - hsv[:, :, 1]) / n

    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 1) * hueMax
    hsv[:, :, 1:3] = np.clip(hsv[:, :, 1:3], 0, 1) * 255
    hsv = hsv.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if (mask is not None):
        mask = mask != 255
        bgr[mask] = np.array([0, 0, 0], dtype=np.uint8)

    return bgr


def load_model(model, modelname):
    preTrainDict = torch.load(modelname)
    model_dict = model.state_dict()
    preTrainDictTemp = {k: v for k, v in preTrainDict.items() if k in model_dict}

    if 0 == len(preTrainDictTemp):
        print("Does not find any module to load. Try DataParallel version.")
        for k, v in preTrainDict.items():
            kk = k[7:]
            if (kk in model_dict):
                preTrainDictTemp[kk] = v

    if (0 == len(preTrainDictTemp)):
        raise Exception("Could not load model from %s." % (modelname), "load_model")

    model_dict.update(preTrainDictTemp)
    model.load_state_dict(model_dict)
    print('Model loaded...')
    return model


def dataset_intrinsics(dataset='tartanair'):
    if dataset == 'kitti':
        focalx, focaly, centerx, centery = 707.0912, 707.0912, 601.8873, 183.1104
    elif dataset == 'euroc':
        focalx, focaly, centerx, centery = 458.6539916992, 457.2959899902, 367.2149963379, 248.3750000000
    elif dataset == 'tartanair':
        focalx, focaly, centerx, centery = 320.0, 320.0, 320.0, 240.0
    elif dataset == 'blur':
        focalx, focaly, centerx, centery = 715.7239956386577, 715.7459592441243, 401.54732887382096, 228.04155560902367
    elif dataset == 'blur_syn':
        focalx, focaly, centerx, centery = 548.409, 548.409, 384.0, 240.0
    elif dataset == 'mbr':
        focalx, focaly, centerx, centery = 981.81817181469557, 981.81817181469557, 320.0, 224.0
    else:
        return None
    return [focalx, focaly, centerx, centery]


def plot_traj(gtposes, estposes, vis=False, savefigname=None, title=''):
    fig = plt.figure(figsize=(4, 4))
    cm = plt.colormaps['Spectral']  # plt.cm.get_cmap is deprecated since matplotlib 3.7

    plt.subplot(111)
    plt.plot(gtposes[:, 0], gtposes[:, 1], linestyle='dashed', c='k')
    plt.plot(estposes[:, 0], estposes[:, 1], c='#ff7f0e')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['Ground Truth', 'TartanVO'])
    plt.title(title)
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)


# def make_intrinsics_layer(w, h, fx, fy, ox, oy):
#     ww, hh = np.meshgrid(range(w), range(h))
#     ww = (ww.astype(np.float32) - ox + 0.5) / fx
#     hh = (hh.astype(np.float32) - oy + 0.5) / fy
#     intrinsicLayer = np.stack((ww, hh)).transpose(1, 2, 0)
#     return intrinsicLayer

def make_intrinsics_layer(w, h, fx, fy, ox, oy, use_gpu=False):
    """
    Build a (2, h, w) intrinsics feature map used as network input.

    Channel 0: (col_index - cx + 0.5) / fx  → normalised x (width) coordinate
    Channel 1: (row_index - cy + 0.5) / fy  → normalised y (height) coordinate

    indexing='ij' is specified explicitly so that:
      hh[i, j] == i  (row / height index)
      ww[i, j] == j  (col  / width  index)
    This is consistent across all PyTorch versions and suppresses the
    UserWarning introduced in PyTorch 1.10.
    """
    if use_gpu:
        hh, ww = torch.meshgrid(
            torch.arange(h).cuda(), torch.arange(w).cuda(), indexing='ij'
        )
    else:
        hh, ww = torch.meshgrid(
            torch.arange(h), torch.arange(w), indexing='ij'
        )

    ww = (ww.float() - ox + 0.5) / fx   # width  axis, normalised by fx / cx
    hh = (hh.float() - oy + 0.5) / fy   # height axis, normalised by fy / cy

    intrinsicLayer = torch.stack((ww, hh), dim=0)   # (2, h, w)
    return intrinsicLayer


def load_kiiti_intrinsics(filename):
    '''
    load intrinsics from kitti intrinsics file
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    cam_intrinsics = lines[2].strip().split(' ')[1:]
    focalx, focaly, centerx, centery = float(cam_intrinsics[0]), float(cam_intrinsics[5]), float(
        cam_intrinsics[2]), float(cam_intrinsics[6])

    return focalx, focaly, centerx, centery


def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=True):
    warmup_epoch = 10 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + math.cos(math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
