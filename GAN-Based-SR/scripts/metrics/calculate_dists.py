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

    img_sr_paths = args.restore

    for dataset in sorted(os.listdir(img_sr_paths)):
        img_sr_path = os.path.join(img_sr_paths, dataset)
        img_gt_path = os.path.join(args.gts, dataset, f"GT/GTmod12")

        dists_list = []

        for img_sr_name in sorted(os.listdir(img_sr_path)):
            img_gt_name = '_'.join(img_sr_name.split('_')[:-1]) + '.png'

            img_sr = prepare_image((Image.open(os.path.join(img_sr_path, img_sr_name)).convert("RGB")))
            img_gt = prepare_image((Image.open(os.path.join(img_gt_path, img_gt_name)).convert("RGB")))

            if args.crop_border != 0:
                img_sr, img_gt = cropborder([img_sr, img_gt], border_size=args.crop_border)

            img_sr = img_sr.to(device)
            img_gt = img_gt.to(device)

            dists = (dists_model(img_gt, img_sr)).item()

            dists_list.append(dists)

        dists_avg = sum(dists_list)/len(dists_list)
        print(f"DISTS for {dataset} is: {dists_avg:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gts', type=str, default='/home/chendu/data2_hdd10t/chendu/dataset/basicsr', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restore', type=str,
                        default='/data1_ssd4t/chendu/myprojects/EFDM_SIM/results/ELANESRGANSSL_bicubic_1_h0.004_ks25_kc9_500_x4_95500/visualization', help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    parser.add_argument('--device', type = int, default=0)
    args = parser.parse_args()
    main(args)