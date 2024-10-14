import cv2
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import scipy.io as io
import argparse
import os
from copy import deepcopy

def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    if args.type == "LoG":
        save_path = os.path.join(args.save_path, os.path.basename(args.gt_path), f"{args.type}/Kernel-{args.Gaussian_kernel_size}/{args.mode}",f"threshold-{args.threshold}")
    else:
        save_path = os.path.join(args.save_path, os.path.basename(args.gt_path), f"{args.type}/{args.mode}", f"threshold-{args.threshold}")
    save_path_mat = os.path.join(save_path, f"mat")
    save_path_png = os.path.join(save_path, f"png")
    os.makedirs(save_path_mat, exist_ok=True)
    os.makedirs(save_path_png, exist_ok=True)

    save_txt_path = os.path.join(save_path, 'statis.txt')
    save_txt = open(save_txt_path, mode='w', encoding='utf-8')

    sum_img = []
    sum_grad = []
    sum_mask = []

    from tqdm import tqdm

    for file in tqdm(list(os.listdir(args.gt_path))):
        img_full_path = os.path.join(args.gt_path, file)
        img_name = os.path.splitext(file)[0]
        img = Image.open(img_full_path).convert(args.mode)
        img = np.array(img)
        if args.mode == "L":
            h, w = img.shape
            c = 1
        elif args.mode == "RGB":
            h,w,c = img.shape
        if args.type == "LoG":
            img = cv2.GaussianBlur(src=img, ksize=(args.Gaussian_kernel_size, args.Gaussian_kernel_size), sigmaX=0, sigmaY=0, dst=-1)
        img_grad = cv2.Laplacian(img, cv2.CV_8U)
        num = h*w*c
        sum_img.append(num)

        num_grad = len(np.where(img_grad>0)[0])
        sum_grad.append(num_grad)

        if args.mode == "L":
            mask = np.zeros((h, w), dtype='int')
        elif args.mode == "RGB":
            mask = np.zeros((h, w, c), dtype='int')
        mask[img_grad>args.threshold] = 1
        num_mask = len(np.where(mask == 1)[0])
        sum_mask.append(num_mask)

        mask_255 = mask * 255
        mask_255 = np.clip(mask_255, 0, 255)
        mask_255 = mask_255.astype(np.uint8)
        mask_255 = np.ascontiguousarray(mask_255)
        mask_255 = Image.fromarray(mask_255)

        img_grad_new = deepcopy(img_grad)
        img_grad_new[img_grad <=args.threshold] = 0
        num_grad_new = len(np.where(img_grad_new>0)[0])

        img_grad_new1 = deepcopy(img_grad)
        img_grad_new1 = img_grad_new1*mask
        num_grad_new1 = len(np.where(img_grad_new1>0)[0])

        assert num_mask == num_grad_new == num_grad_new1, f"The mask number-{num_mask}, grad new number-{num_grad_new}, grad new1 number-{num_grad_new1} are not the same."

        save_txt.write(f"{img_name}:\n")
        save_txt.write(f"Image number-{num}, grad number-{num_grad}-{num_grad/num:.4f}, mask number-{num_mask}-{num_mask/num:.4f}\n\n")

        mask_255.save(os.path.join(save_path_png, f"{img_name}.png"), "PNG", quality = 95)
        io.savemat(os.path.join(save_path_mat, f"{img_name}.mat"), {'mat':mask}, do_compression=True)

    print(f"Maximum of grad is {max(sum_grad):.2f}, percentage is {max(sum_grad)/(h*w*c):4f}")
    print(f"Minium of grad is {min(sum_grad):.2f}, percentage is {min(sum_grad)/(h*w*c):.4f}")
    print(f"Average of grad is {sum(sum_grad)/len(sum_grad):.2f}, percentage is {sum(sum_grad)/(sum(sum_img)):.4f}")
    print(f"Maximum of mask is {max(sum_mask):.2f}, percentage is {max(sum_mask)/(h*w*c):4f}")
    print(f"Minium of grad is {min(sum_mask):.2f}, percentage is {min(sum_mask)/(h*w*c):.4f}")
    print(f"Average of grad is {sum(sum_mask)/len(sum_mask):.2f}, percentage is {sum(sum_mask)/(sum(sum_img)):.4f}")

    save_txt.write(f"Maximum of grad is {max(sum_grad):.2f}, percentage is {max(sum_grad)/(h*w*c):4f}\n")
    save_txt.write(f"Minium of grad is {min(sum_grad):.2f}, percentage is {min(sum_grad)/(h*w*c):.4f}\n")
    save_txt.write(f"Average of grad is {sum(sum_grad)/len(sum_grad):.2f}, percentage is {sum(sum_grad)/(sum(sum_img)):.4f}\n")
    save_txt.write(f"Maximum of mask is {max(sum_mask):.2f}, percentage is {max(sum_mask)/(h*w*c):4f}\n")
    save_txt.write(f"Minium of grad is {min(sum_mask):.2f}, percentage is {min(sum_mask)/(h*w*c):.4f}\n")
    save_txt.write(f"Average of grad is {sum(sum_mask)/len(sum_mask):.2f}, percentage is {sum(sum_mask)/(sum(sum_img)):.4f}\n")

    save_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='/home/chendu/data2_hdd10t/chendu/dataset/basicsr/DIV2K100/GT/GTmod12', help='Path to gt (Ground-Truth)')
    #/home/chendu/data2_hdd10t/chendu/dataset/basicsr/BSDS100/GTmod24
    #/home/chendu/data2_hdd10t/chendu/dataset/DF2K_OST/multiscale_HR_sub_512
    parser.add_argument('--save_path', type=str, default='/home/chendu/data2_hdd10t/chendu/dataset/basicsr/DIV2K100/mask_selfsim', help='Path to save path')
    #/home/chendu/data2_hdd10t/chendu/dataset/basicsr/BSDS100/mask_selfsim
    #/home/chendu/data2_hdd10t/chendu/dataset/DF2K_OST/mask_selfsim
    parser.add_argument("--threshold", type=float, default = 20.0)
    parser.add_argument("--mode", type=str, default="L")
    parser.add_argument("--Gaussian_kernel_size", type = int, default=5)
    parser.add_argument("--type", type=str, default="Laplacian")
    args = parser.parse_args()
    main(args)
