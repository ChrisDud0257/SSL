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
    device = torch.device(f"cuda:{args.device}")

    dists_model = DISTS().to(device)

    for idx, dataset in enumerate(args.gt):
        save_txt_path = os.path.join(os.path.dirname(args.restored[idx]), f"DISTS_{os.path.basename(os.path.dirname(dataset))}.txt")
        save_txt = open(save_txt_path, mode='w', encoding='utf-8')

        img_list = sorted(glob.glob(osp.join(dataset, '*')))

        dists_list = []

        for i, img_path in enumerate(img_list):
            basename, ext = osp.splitext(osp.basename(img_path))

            img_sr = prepare_image((Image.open(osp.join(args.restored[idx], basename + args.suffix + ext)).convert("RGB")))
            img_gt = prepare_image((Image.open(img_path).convert("RGB")))

            if args.crop_border != 0:
                img_sr, img_gt = cropborder([img_sr, img_gt], border_size=args.crop_border)

            img_sr = img_sr.to(device)
            img_gt = img_gt.to(device)

            dists = (dists_model(img_gt, img_sr)).item()

            print(f'{i+1:3d}: {basename:25}. \tDISTS: {dists:.6f}.')
            save_txt.write(f"{basename:25}. \tDISTS: {dists:.6f}.\n")

            dists_list.append(dists)

        dists_avg = sum(dists_list)/len(dists_list)
        print(f"DISTS for {dataset} is: {dists_avg:.6f}")
        save_txt.write(f"DISTS for {dataset} is: {dists_avg:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gt', type=str, default='/home/chendu/data2_hdd10t/chendu/dataset/DIV2K/DIV2K_valid_HR_mod24', help='Path to gt (Ground-Truth)')
    parser.add_argument('--gt', nargs='+', default=[
        '/data1_ssd4t/chendu/datasets/StableSR/StableSR-TestSets/StableSR_testsets/DIV2K_V2_val/gt',
        '/data1_ssd4t/chendu/datasets/StableSR/StableSR-TestSets/StableSR_testsets/DrealSRVal_crop128/gt',
        '/data1_ssd4t/chendu/datasets/StableSR/StableSR-TestSets/StableSR_testsets/RealSRVal_crop128/gt'],
                        help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', nargs='+', default=[
        '/data1_ssd4t/chendu/datasets/StableSR/StableSR-TestSets/StableSR_testsets/StableSR_w0.5_results/V2_T1000_S200_117_W0.5/samples',
        '/data1_ssd4t/chendu/datasets/StableSR/StableSR-TestSets/StableSR_testsets/StableSR_w0.5_results/V2_T1000_S200_117_W0.5_DrealSR/samples',
        '/data1_ssd4t/chendu/datasets/StableSR/StableSR-TestSets/StableSR_testsets/StableSR_w0.5_results/V2_T1000_S200_117_W0.5_RealSR/samples'],
                        help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    main(args)