import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os


class DatasetMultiLROneGT(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetMultiLROneGT, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf

        self.gt_folder = opt['dataroot_H']
        self.lq_folder = opt['dataroot_L']

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.img_name_list = [i for i in os.listdir(self.lq_folder)]

    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        gt_name = f'{os.path.splitext(img_name)[0].split("+")[0]}.png'
        lq_path = os.path.join(self.lq_folder, img_name)
        gt_path = os.path.join(self.gt_folder, gt_name)

        img_H = util.imread_uint(gt_path, self.n_channels)
        img_H = util.uint2single(img_H)

        img_H = util.modcrop(img_H, self.sf)

        img_L = util.imread_uint(lq_path, self.n_channels)
        img_L = util.uint2single(img_L)


        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)


        return {'L': img_L, 'H': img_H, 'L_path': lq_path, 'H_path': gt_path}

    def __len__(self):
        return len(self.img_name_list)
