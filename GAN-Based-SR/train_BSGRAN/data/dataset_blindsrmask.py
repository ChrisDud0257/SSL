import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os
from utils import utils_blindsr as blindsr
import scipy.io as io


class DatasetBlindSRMask(data.Dataset):
    '''
    # -----------------------------------------
    # dataset for BSRGAN
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(DatasetBlindSRMask, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.shuffle_prob = opt['shuffle_prob'] if opt['shuffle_prob'] else 0.1
        self.use_sharp = opt['use_sharp'] if opt['use_sharp'] else False
        self.degradation_type = opt['degradation_type'] if opt['degradation_type'] else 'bsrgan'
        self.lq_patchsize = self.opt['lq_patchsize'] if self.opt['lq_patchsize'] else 64
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else self.lq_patchsize*self.sf

        self.paths_H = util.get_image_paths(opt['dataroot_H'])

        self.paths_H_mask = opt['dataroot_H_mask']
        print(len(self.paths_H))

#        for n, v in enumerate(self.paths_H):
#            if 'face' in v:
#                del self.paths_H[n]
#        time.sleep(1)
        assert self.paths_H, 'Error: H path is empty.'

    def __getitem__(self, index):

        L_path = None

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_name, ext = os.path.splitext(os.path.basename(H_path))


        H_mask = io.loadmat(os.path.join(self.paths_H_mask, f"{img_name}.mat"))['mat']
        H_mask = np.array(H_mask).astype(np.float32)
        H_mask = np.ascontiguousarray(H_mask) #h*w, 0,1, ndarry
        if H_mask.ndim == 2 :
            H_mask = np.expand_dims(H_mask, axis=2) #h*w*1, 0,1, ndarry

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_H.shape

            rnd_h_H = random.randint(0, max(0, H - self.patch_size))
            rnd_w_H = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            H_mask = H_mask[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            if 'face' in img_name:
                mode = random.choice([0, 4])
                img_H = util.augment_img(img_H, mode=mode)
                H_mask = util.augment_img(H_mask)
            else:
                mode = random.randint(0, 7)
                img_H = util.augment_img(img_H, mode=mode)
                H_mask = util.augment_img(H_mask)

            img_H = util.uint2single(img_H)
            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        else:
            img_H = util.uint2single(img_H)
            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        H_mask = util.single2tensor3(H_mask)

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path,
                'H_mask': H_mask}

    def __len__(self):
        return len(self.paths_H)
