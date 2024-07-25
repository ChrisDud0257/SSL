from torch.utils import data as data
from torchvision.transforms.functional import normalize
import os
import scipy.io as io
import numpy as np

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_img_mask
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor, imfromfile, imfromfile255
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class MultiLROneGTDataset(data.Dataset):

    def __init__(self, opt):
        super(MultiLROneGTDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        self.img_name_list = [i for i in os.listdir(self.lq_folder)]

        self.img_range = opt.get('img_range', 1)

    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        gt_name = f'{os.path.splitext(img_name)[0].split("+")[0]}.png'
        lq_path = os.path.join(self.lq_folder, img_name)
        gt_path = os.path.join(self.gt_folder, gt_name)
        if self.img_range == 1:
            img_gt = imfromfile(path=gt_path, float32=True)  #h*w*c, 0-1, ndarray
            img_lq = imfromfile(path=lq_path, float32=True)  #h*w*c, 0-1, ndarray
        else:
            img_gt = imfromfile255(path=gt_path, float32=True)  #h*w*c, 0-255, ndarray
            img_lq = imfromfile255(path=lq_path, float32=True)  #h*w*c, 0-255, ndarray


        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)


        return_d = {'gt':img_gt, 'lq':img_lq , 'lq_path': lq_path, 'gt_path': gt_path}

        return return_d

    def __len__(self):
        return len(self.img_name_list)
