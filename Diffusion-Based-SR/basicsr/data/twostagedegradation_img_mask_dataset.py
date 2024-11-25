import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from torch.utils import data as data
import json
import scipy.io as io

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, imfromfile
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class TwoStageDegradation_Img_Mask_Dataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(TwoStageDegradation_Img_Mask_Dataset, self).__init__()
        self.opt = opt

        self.gt_mask_folder = opt['dataroot_gt_mask']

        all_img_path_list = []
        for dataset in opt['gt_path']:
            for image in sorted(os.listdir(dataset)):
                all_img_path_list.append(os.path.join(dataset, image))

        all_img_mask_path_list = []
        for dataset_gt_mask in opt['dataroot_gt_mask']:
            for mask in sorted(os.listdir(dataset_gt_mask)):
                all_img_mask_path_list.append(os.path.join(dataset_gt_mask, mask))

        if 'face_gt_path' in opt:
            self.face_gt_path = opt['face_gt_path']
            self.all_face_gt_list = [i for i in sorted(os.listdir(self.face_gt_path))]
            random.shuffle(self.all_face_gt_list)
            self.choose_face_gt_list = self.all_face_gt_list[:opt['num_face']]
            for image in self.choose_face_gt_list:
                all_img_path_list.append(os.path.join(self.face_gt_path, image))
            self.choose_face_gt_mask_list = [f'{os.path.splitext(i)[0]}.mat' for i in self.choose_face_gt_list]
            for mask in self.choose_face_gt_mask_list:
                all_img_mask_path_list.append(os.path.join(opt['dataroot_face_gt_mask'], mask))

        for index, i in enumerate(all_img_path_list):
            assert os.path.splitext(os.path.basename(i))[0] == os.path.splitext(os.path.basename(all_img_mask_path_list[index]))[0], \
            f'The image in gt_path-{os.path.splitext(os.path.basename(i))[0]} is not the same as the mask in ' \
            f'mask_path-{os.path.splitext(os.path.basename(all_img_mask_path_list[index]))[0]}'

        # self.suitable_img_gt_path, self.suitable_img_mask_path = self.obtain_img_with_suitable_size(all_img_path_list=all_img_path_list,
        #                                                                                             all_img_mask_path_list=all_img_mask_path_list)
        self.suitable_img_gt_path = all_img_path_list
        self.suitable_img_mask_path = all_img_mask_path_list

        # blur settings for the first degradation
        self.blur_kernel_size_min = opt['blur_kernel_size_min']
        self.blur_kernel_size_max = opt['blur_kernel_size_max']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size_min2 = opt['blur_kernel_size_min2']
        self.blur_kernel_size_max2 = opt['blur_kernel_size_max2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']  # a list for each kernel probability
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']  # betag used in generalized Gaussian blur kernels
        self.betap_range2 = opt['betap_range2']  # betap used in plateau blur kernels
        self.sinc_prob2 = opt['sinc_prob2']  # the probability for sinc filters

        self.kernel_range = [2 * v + 1 for v in range(self.blur_kernel_size_min, self.blur_kernel_size_max + 1)]  # kernel size ranges from 3 to 9
        self.kernel_range2 = [2 * v + 1 for v in range(self.blur_kernel_size_min2,
                                                      self.blur_kernel_size_max2 + 1)]  # kernel size ranges from 3 to 9

        self.final_sinc_prob = opt['final_sinc_prob']


        self.pulse_tensor = torch.zeros(self.kernel_range[-1], self.kernel_range[-1]).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[self.kernel_range[-1] // 2, self.kernel_range[-1] // 2] = 1

        self.crop_size = opt.get('crop_size', 512)

    def obtain_img_with_suitable_size(self, all_img_path_list, all_img_mask_path_list):
        suitable_img_gt_path = []
        suitable_img_mask_path = []
        for index, gt_path in enumerate(all_img_path_list):
            img_gt = imfromfile(path=gt_path, float32=True)
            print(f'The gt path now is {gt_path}')
            h,w,_ = img_gt.shape
            if h>=512 and w>= 512:
                suitable_img_gt_path.append(gt_path)
                suitable_img_mask_path.append(all_img_mask_path_list[index])

        return suitable_img_gt_path, suitable_img_mask_path

    def __getitem__(self, index):
        gt_path = self.suitable_img_gt_path[index]

        img_gt = imfromfile(path=gt_path, float32=True)


        gt_mask = io.loadmat(self.suitable_img_mask_path[index])['mat']
        gt_mask = np.array(gt_mask).astype(np.float32)
        gt_mask = np.ascontiguousarray(gt_mask) #h*w, 0,1, ndarry
        if gt_mask.ndim == 2 :
            gt_mask = np.expand_dims(gt_mask, axis=2) #h*w*1, 0,1, ndarry

        # print(f"gt mask is {gt_mask}")

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt, gt_mask = augment([img_gt, gt_mask], self.opt['use_hflip'], self.opt['use_rot'])

        h, w = img_gt.shape[0:2]
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        img_gt = img_gt[top:top + self.crop_size, left:left + self.crop_size, ...]
        gt_mask = gt_mask[top:top + self.crop_size, left:left + self.crop_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (self.kernel_range[-1] - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (self.kernel_range[-1] - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=self.kernel_range[-1])
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, gt_mask = img2tensor([img_gt, gt_mask], bgr2rgb=True, float32=True)

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)


        return_d = {'gt': img_gt,
                    'kernel1': kernel,'kernel2': kernel2,
                    'sinc_kernel': sinc_kernel,
                    'gt_mask':gt_mask}
        return return_d

    def __len__(self):
        return len(self.suitable_img_gt_path)
