import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
import torch
import os
from collections import Counter
import argparse
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F
from DISTS_pytorch import DISTS
import argparse
from PIL import Image

def prepare_image(image, resize=False):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

def cropborder(imgs, border_size = 0):
    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [i[:, :, border_size:-border_size, border_size:-border_size] for i in imgs]
    if len(imgs) == 0:
        return imgs[0]
    else:
        return imgs

def main(args):
    device = torch.device(f"cuda")
    dists_model = DISTS().to(device)

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

                save_txt_path = os.path.join(os.path.dirname(os.path.dirname(dataset_path)), f"DISTS_{dataset}.txt")
                save_txt = open(save_txt_path, mode='w', encoding='utf-8')

                dists_list = []

                print(f"----------------------------")
                print(f"Now the testing dataset is {dataset_path}")

                img_path_lq_list = [os.path.join(dataset_path, i) for i in os.listdir(dataset_path)]
                for i, img_path_lq in enumerate(img_path_lq_list):
                    img_lq_name, ext = osp.splitext(osp.basename(img_path_lq))
                    img_gt_name = ('_'.join(img_lq_name.split('_')[:-1])).split('+')

                    img_sr = prepare_image((Image.open(img_path_lq)).convert("RGB"))
                    img_gt = prepare_image((Image.open(osp.join(gt_path, img_gt_name+ext)).convert("RGB")))

                    if args.crop_border != 0:
                        img_sr, img_gt = cropborder([img_sr, img_gt], border_size=args.crop_border)

                    img_sr = img_sr.to(device)
                    img_gt = img_gt.to(device)

                    dists = (dists_model(img_gt, img_sr)).item()

                    print(f'{i + 1:3d}: {img_lq_name:25}. \tDISTS: {dists:.6f}.')
                    save_txt.write(f"{img_lq_name:25}. \tDISTS: {dists:.6f}.\n")

                    dists_list.append(dists)
                dists_avg = sum(dists_list) / len(dists_list)
                print(f"Average DISTS for {dataset} is: {dists_avg:.6f}")
                print(f"----------------------------")

                save_txt.write(f"Average DISTS for {dataset}: {dists_avg:.6f}")
                save_txt.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gt', type=str, default='/home/chendu/data2_hdd10t/chendu/dataset/DIV2K/DIV2K_valid_HR_mod24', help='Path to gt (Ground-Truth)')
    parser.add_argument('--gt', nargs='+', default=['/home/chendu/data2_hdd10t/chendu/dataset/basicsr'],
                        help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', nargs='+', default=['/home/chendu/data2_hdd10t/chendu/myprojects/ISSL_Results/Real-GAN'], help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    args = parser.parse_args()
    main(args)