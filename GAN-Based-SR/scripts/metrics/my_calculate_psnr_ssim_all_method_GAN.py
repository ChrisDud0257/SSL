import argparse
import os.path

import cv2
import numpy as np
from os import path as osp

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import bgr2ycbcr, scandir


def main(args):
    """Calculate PSNR and SSIM for images.
    """
    for idx, mtd in enumerate(args.restored):

        methods = os.listdir(mtd)

        for method in methods:
            dataset_path_all_one_method = os.path.join(mtd, method, 'visualization')
            dataset_paths = os.listdir(dataset_path_all_one_method)

            if 'DPEDiphoneValSet_crop128' in dataset_paths:
                dataset_paths.remove('DPEDiphoneValSet_crop128')
            for dataset in dataset_paths:
                dataset_path = os.path.join(dataset_path_all_one_method, dataset)

                gt_path = os.path.join(args.gt[idx], dataset, 'GT', 'GTmod12')
                psnr_all = []
                ssim_all = []

                save_txt_path = os.path.join(os.path.dirname(os.path.dirname(dataset_path)), f"PSNR_SSIM_{dataset}.txt")
                save_txt = open(save_txt_path, mode='w', encoding='utf-8')
                print(f"----------------------------")
                print(f"Now the testing dataset is {dataset_path}")
                if args.test_y_channel:
                    print('Testing Y channel.')
                else:
                    print('Testing RGB channels.')

                img_path_lq_list = [os.path.join(dataset_path, i) for i in os.listdir(dataset_path)]
                for i, img_path_lq in enumerate(img_path_lq_list):
                    img_lq_name, ext = osp.splitext(osp.basename(img_path_lq))
                    img_gt_name = '_'.join(img_lq_name.split('_')[:-1])
                    img_restored = cv2.imread(img_path_lq, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
                    img_gt = cv2.imread(osp.join(gt_path, img_gt_name+ext), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.


                    if args.correct_mean_var:
                        mean_l = []
                        std_l = []
                        for j in range(3):
                            mean_l.append(np.mean(img_gt[:, :, j]))
                            std_l.append(np.std(img_gt[:, :, j]))
                        for j in range(3):
                            # correct twice
                            mean = np.mean(img_restored[:, :, j])
                            img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                            std = np.std(img_restored[:, :, j])
                            img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                            mean = np.mean(img_restored[:, :, j])
                            img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                            std = np.std(img_restored[:, :, j])
                            img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                    if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
                        img_gt = bgr2ycbcr(img_gt, y_only=True)
                        img_restored = bgr2ycbcr(img_restored, y_only=True)

                    # calculate PSNR and SSIM
                    psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
                    ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
                    print(f'{i+1:3d}: {img_lq_name:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}')
                    save_txt.write(f"{img_lq_name:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}\n")
                    psnr_all.append(psnr)
                    ssim_all.append(ssim)
                print(f'Average PSNR for {dataset}: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}')
                save_txt.write(f'Average PSNR for {dataset}: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}')

                save_txt.close()
                print(f"----------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gt', type=str, default='/home/chendu/data2_hdd10t/chendu/dataset/DIV2K/DIV2K_valid_HR_mod24', help='Path to gt (Ground-Truth)')
    parser.add_argument('--gt', nargs='+', default=['/home/chendu/data2_hdd10t/chendu/dataset/basicsr'],
                        help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', nargs='+', default=['/home/chendu/data2_hdd10t/chendu/myprojects/ISSL_Results/GAN'], help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    parser.add_argument(
        '--test_y_channel',
        type=bool,
        default=True,
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    args = parser.parse_args()
    main(args)
