import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
import argparse
import os
from basicsr.utils import img2tensor

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def main(args):

    for idx, dataset in enumerate(args.gt):
        save_txt_path = os.path.join(os.path.dirname(args.restored[idx]), f"LPIPS_{os.path.basename(os.path.dirname(dataset))}.txt")
        save_txt = open(save_txt_path, mode='w', encoding='utf-8')
        # -------------------------------------------------------------------------
        loss_fn_vgg = lpips.LPIPS(net='alex').cuda()  # RGB, normalized to [-1,1]
        lpips_all = []
        img_list = sorted(glob.glob(osp.join(dataset, '*')))

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        crop_border = args.crop_border
        for i, img_path in enumerate(img_list):
            basename, ext = osp.splitext(osp.basename(img_path))
            img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            img_restored = cv2.imread(osp.join(args.restored[idx], basename + args.suffix + ext), cv2.IMREAD_UNCHANGED).astype(
                np.float32) / 255.

            if crop_border != 0:
                img_gt = img_gt[crop_border:-crop_border, crop_border:-crop_border, ...]
                img_restored = img_restored[crop_border:-crop_border, crop_border:-crop_border, ...]


            img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
            # norm to [-1, 1]
            normalize(img_gt, mean, std, inplace=True)
            normalize(img_restored, mean, std, inplace=True)

            # calculate lpips
            lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())

            print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val.item():.6f}.')
            save_txt.write(f"{basename:25}. \tLPIPS: {lpips_val.item():.6f}.\n")
            lpips_all.append(lpips_val.item())

        print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')
        save_txt.write(f"Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}")
        save_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gt', type=str, default='/home/chendu/data2_hdd10t/chendu/dataset/DIV2K/DIV2K_valid_HR_mod24', help='Path to gt (Ground-Truth)')
    parser.add_argument('--gt', nargs='+', default=['/home/chendu/data2_hdd10t/chendu/dataset/basicsr/DIV2K100/GT/GTmod12'],
                        help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', nargs='+', default=['/home/chendu/data2_hdd10t/chendu/FinalUpload/ACMMM2024FinalUpload/SSL/GAN-Based-SR/results/ESRGANSSL_bicubic_x4/visualization/DIV2K100'],
                        help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    parser.add_argument('--suffix', type=str, default='_ESRGANSSLbicubicx4', help='Suffix for restored images')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    args = parser.parse_args()
    main(args)
