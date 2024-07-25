import torch
from pyiqa.archs.musiq_arch import MUSIQ
import os
import numpy as np
import cv2
from PIL import Image
import argparse
from basicsr.utils import img2tensor
import os.path as osp
import glob
import pyiqa

def cropborder(img, crop_size = 4):
    img_new = img[crop_size:-crop_size, crop_size:-crop_size, :]
    return img_new


def main(args):
    musiq = pyiqa.create_metric('musiq', pretrained_model_path=None).cuda()

    for idx, dataset in enumerate(args.input):
        name = os.path.basename(dataset)
        img_list = sorted(glob.glob(osp.join(dataset, '*')))

        save_txt_path = os.path.join(os.path.dirname(os.path.dirname(dataset)), f"MUSIQ_{name}.txt")
        save_txt = open(save_txt_path, mode='w', encoding='utf-8')
        musiq_score_all = []

        for i, img_path in enumerate(img_list):
            basename, _ = os.path.splitext(os.path.basename(img_path))

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            img = cropborder(img, crop_size=args.crop_border)

            img = img2tensor(img, bgr2rgb=True, float32=True).cuda()

            musiq_score = musiq(img).item()
            print(f'{i + 1:3d}: {basename:25}. \tMUSIQ: {musiq_score}')
            save_txt.write(f"{basename}. \tMUSIQ: {musiq_score:.6f}\n")

            musiq_score_all.append(musiq_score)

        print(f"Average MUSIQ for {name}: {sum(musiq_score_all) / len(musiq_score_all)}")

        save_txt.write(f"Average MUSIQ for {name}: {sum(musiq_score_all) / len(musiq_score_all):.6f}")
        save_txt.close()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--input", nargs='+', default=[
        "/home/chendu/data2_hdd10t/chendu/myprojects/ISSL_Results/DM/DiffIRS2_GANSSL_x4_390000/visualization/DrealSRVal_crop128"])
    parse.add_argument("--crop_border", type=int, default=4)
    args = parse.parse_args()
    main(args)