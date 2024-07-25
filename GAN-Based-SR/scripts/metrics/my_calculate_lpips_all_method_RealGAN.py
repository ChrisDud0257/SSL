import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
import argparse
import os
import lpips
from basicsr.utils import img2tensor

def main(args):
    crop_border = args.crop_border

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

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

                save_txt_path = os.path.join(os.path.dirname(os.path.dirname(dataset_path)), f"LPIPS_{dataset}.txt")
                save_txt = open(save_txt_path, mode='w', encoding='utf-8')

                loss_fn_vgg = lpips.LPIPS(net='alex').cuda()  # RGB, normalized to [-1,1]
                lpips_all = []

                print(f"----------------------------")
                print(f"Now the testing dataset is {dataset_path}")

                img_path_lq_list = [os.path.join(dataset_path, i) for i in os.listdir(dataset_path)]
                for i, img_path_lq in enumerate(img_path_lq_list):
                    img_lq_name, ext = osp.splitext(osp.basename(img_path_lq))
                    img_gt_name = ('_'.join(img_lq_name.split('_')[:-1])).split('+')[0]
                    img_restored = cv2.imread(img_path_lq, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
                    img_gt = cv2.imread(osp.join(gt_path, img_gt_name+ext), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

                    if crop_border != 0:
                        img_gt = img_gt[crop_border:-crop_border, crop_border:-crop_border, ...]
                        img_restored = img_restored[crop_border:-crop_border, crop_border:-crop_border, ...]

                    img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
                    # norm to [-1, 1]
                    normalize(img_gt, mean, std, inplace=True)
                    normalize(img_restored, mean, std, inplace=True)

                    # calculate lpips
                    lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())

                    print(f'{i + 1:3d}: {img_lq_name:25}. \tLPIPS: {lpips_val.item():.6f}.')
                    save_txt.write(f"{img_lq_name:25}. \tLPIPS: {lpips_val.item():.6f}.\n")
                    lpips_all.append(lpips_val.item())

                print(f'Average LPIPS for {dataset}: {sum(lpips_all) / len(lpips_all):.6f}')
                save_txt.write(f"Average LPIPS for {dataset}: {sum(lpips_all) / len(lpips_all):.6f}")
                save_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gt', type=str, default='/home/chendu/data2_hdd10t/chendu/dataset/DIV2K/DIV2K_valid_HR_mod24', help='Path to gt (Ground-Truth)')
    parser.add_argument('--gt', nargs='+', default=['/home/chendu/data2_hdd10t/chendu/dataset/basicsr'],
                        help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', nargs='+', default=['/home/chendu/data2_hdd10t/chendu/myprojects/ISSL_Results/Real-GAN'], help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    args = parser.parse_args()
    main(args)