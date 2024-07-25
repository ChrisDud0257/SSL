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

    for idx, mtd in enumerate(args.input):
        methods = os.listdir(mtd)
        for method in methods:
            dataset_path_all_one_method = os.path.join(mtd, method, 'visualization')
            dataset_paths = os.listdir(dataset_path_all_one_method)
            for dataset in dataset_paths:
                name = os.path.basename(dataset)
                dataset_path = os.path.join(dataset_path_all_one_method, dataset)

                save_txt_path = os.path.join(os.path.dirname(os.path.dirname(dataset_path)), f"CLIPIQA_{dataset}.txt")
                save_txt = open(save_txt_path, mode='w', encoding='utf-8')

                clipiqa_score_all = []
                img_list = sorted(glob.glob(osp.join(dataset_path, '*')))

                print(f"----------------------------")
                print(f"Now the testing dataset is {dataset_path}")
                # print(f"img list is {img_list}")
                for i, img_path in enumerate(img_list):
                    basename, _ = os.path.splitext(os.path.basename(img_path))

                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
                    img = cropborder(img, crop_size=args.crop_border)

                    img = img2tensor(img, bgr2rgb=True, float32=True)

                    clipiqa_score = clipiqa(img, device=device).item()
                    print(f'{i + 1:3d}: {basename:25}. \tCLIPIQA: {clipiqa_score:.6f}')
                    save_txt.write(f"{basename}. \tCLIPIQA: {clipiqa_score:.6f}\n")

                    clipiqa_score_all.append(clipiqa_score)

                print(f"Average CLIPIQA for {name}: {sum(clipiqa_score_all) / len(clipiqa_score_all):.6f}")

                save_txt.write(f"Average CLIPIQA for {name}: {sum(clipiqa_score_all) / len(clipiqa_score_all):.6f}")
                save_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs = '+', default=['/home/chendu/data2_hdd10t/chendu/myprojects/ISSL_Results/Real-GAN'], help='Input path')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--backbone_path", type=str, default="/data1_ssd4t/chendu/myprojects/EFDM_SIM/experiments/pretrained_models/clipmodels/RN50.pt")
    args = parser.parse_args()
    main(args)
