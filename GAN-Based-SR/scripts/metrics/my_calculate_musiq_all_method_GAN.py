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
import glob

def cropborder(img, crop_size = 4):
    img_new = img[crop_size:-crop_size, crop_size:-crop_size, :]
    return img_new


def main(args):
    musiq = pyiqa.create_metric('musiq', pretrained_model_path=None).cuda()

    for idx, mtd in enumerate(args.input):
        methods = os.listdir(mtd)
        for method in methods:
            dataset_path_all_one_method = os.path.join(mtd, method, 'visualization')
            dataset_paths = os.listdir(dataset_path_all_one_method)
            for dataset in dataset_paths:
                name = os.path.basename(dataset)
                dataset_path = os.path.join(dataset_path_all_one_method, dataset)

                save_txt_path = os.path.join(os.path.dirname(os.path.dirname(dataset_path)), f"MUSIQ_{dataset}.txt")
                save_txt = open(save_txt_path, mode='w', encoding='utf-8')

                musiq_score_all = []
                img_list = sorted(glob.glob(osp.join(dataset_path, '*')))

                print(f"----------------------------")
                print(f"Now the testing dataset is {dataset_path}")
                # print(f"img list is {img_list}")
                for i, img_path in enumerate(img_list):
                    basename, _ = os.path.splitext(os.path.basename(img_path))

                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
                    img = cropborder(img, crop_size=args.crop_border)

                    img = img2tensor(img, bgr2rgb=True, float32=True)

                    musiq_score = musiq(img).item()
                    print(f'{i + 1:3d}: {basename:25}. \tMUSIQ: {musiq_score:.6f}')
                    save_txt.write(f"{basename}. \tMUSIQ: {musiq_score:.6f}\n")

                    musiq_score_all.append(musiq_score)

                print(f"Average MUSIQ for {name}: {sum(musiq_score_all) / len(musiq_score_all):.6f}")

                save_txt.write(f"Average MUSIQ for {name}: {sum(musiq_score_all) / len(musiq_score_all):.6f}")
                save_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs = '+', default=['/home/chendu/data2_hdd10t/chendu/myprojects/ISSL_Results/GAN'], help='Input path')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    main(args)
