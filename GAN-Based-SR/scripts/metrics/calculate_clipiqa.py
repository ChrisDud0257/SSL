from clipiqa_pyiqa.clipiqa_arch import CLIPIQA
import glob
import os
import torch
import torch.nn as nn
import argparse
from PIL import Image
import cv2
import numpy as np
from basicsr.utils import img2tensor
import os.path as osp



def cropborder(img, crop_size = 4):
    img_new = img[crop_size:-crop_size, crop_size:-crop_size, :]
    return img_new

def main(args):
    device = torch.device(f"cuda:{args.device}")
    clipiqa = CLIPIQA(backbone=args.backbone_path)

    for idx, dataset in enumerate(args.input):
        name = os.path.basename(dataset)
        img_list = sorted(glob.glob(osp.join(dataset, '*')))

        save_txt_path = os.path.join(os.path.dirname(os.path.dirname(dataset)), f"CLIPIQA_{name}.txt")
        save_txt = open(save_txt_path, mode='w', encoding='utf-8')
        clipiqa_score_all = []

        for i, img_path in enumerate(img_list):
            basename, _ = os.path.splitext(os.path.basename(img_path))

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            img = cropborder(img, crop_size=args.crop_border)

            img = img2tensor(img, bgr2rgb=True, float32=True)

            clipiqa_score = clipiqa(img, device = device).item()
            print(f'{i + 1:3d}: {basename:25}. \tCLIPIQA: {clipiqa_score:.6f}')
            save_txt.write(f"{basename}. \tCLIPIQA: {clipiqa_score:.6f}\n")

            clipiqa_score_all.append(clipiqa_score)

        print(f"Average CLIPIQA for {name}: {sum(clipiqa_score_all) / len(clipiqa_score_all):.6f}")

        save_txt.write(f"Average CLIPIQA for {name}: {sum(clipiqa_score_all) / len(clipiqa_score_all):.6f}")
        save_txt.close()



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--input", nargs = '+', default=['/home/chendu/data2_hdd10t/chendu/FinalUpload/ACMMM2024FinalUpload/SSL/GAN-Based-SR/results/ESRGANSSL_bicubic_x4/visualization/DIV2K100'])
    parse.add_argument("--crop_border", type=int, default=4)
    parse.add_argument("--device", type=int, default=0)
    parse.add_argument("--backbone_path", type=str, default="pretrained_models/clipmodels/RN50.pt")
    args = parse.parse_args()
    main(args)
