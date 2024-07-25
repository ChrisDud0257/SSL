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
class PairedImageMaskDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageMaskDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.gt_mask_folder = opt['dataroot_gt_mask']

        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.img_name_list = [i for i in os.listdir(self.gt_folder)]

        self.img_range = opt.get('img_range', 1)

    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        if self.img_range == 1:
            img_gt = imfromfile(path=os.path.join(self.gt_folder, img_name), float32=True)  #h*w*c, 0-1, ndarray
            img_lq = imfromfile(path=os.path.join(self.lq_folder, img_name), float32=True)  #h*w*c, 0-1, ndarray
        else:
            img_gt = imfromfile255(path=os.path.join(self.gt_folder, img_name), float32=True)  #h*w*c, 0-255, ndarray
            img_lq = imfromfile255(path=os.path.join(self.lq_folder, img_name), float32=True)  #h*w*c, 0-255, ndarray

        scale = self.opt['scale']

        gt_mask = io.loadmat(os.path.join(self.gt_mask_folder, f"{os.path.splitext(img_name)[0]}.mat"))['mat']
        gt_mask = np.array(gt_mask).astype(np.float32)
        gt_mask = np.ascontiguousarray(gt_mask) #h*w, 0,1, ndarry
        if gt_mask.ndim == 2 :
            gt_mask = np.expand_dims(gt_mask, axis=2) #h*w*1, 0,1, ndarry


        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq, gt_mask = paired_random_crop_img_mask(img_gts=img_gt, img_lqs=img_lq,
                                                                  masks=gt_mask, gt_patch_size=gt_size, scale=scale)
            img_gt, img_lq, gt_mask = augment([img_gt, img_lq, gt_mask], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, gt_mask = img2tensor([img_gt, img_lq, gt_mask], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return_d = {'gt':img_gt, 'lq':img_lq, 'gt_mask':gt_mask}

        return return_d

    def __len__(self):
        return len(self.img_name_list)
