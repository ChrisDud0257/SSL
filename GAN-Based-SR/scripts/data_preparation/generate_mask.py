import cv2
from PIL import Image
import numpy as np
import scipy.io as io
import argparse
import os

def main(args):
    dataset = args.input
    save_path = args.save
    os.makedirs(save_path, exist_ok=True)
    save_path_mat = os.path.join(save_path, f"mat")
    save_path_png = os.path.join(save_path, f"png")
    os.makedirs(save_path_mat, exist_ok=True)
    os.makedirs(save_path_png, exist_ok=True)

    for file in os.listdir(dataset):
        img_full_path = os.path.join(dataset, file)
        img_name = os.path.splitext(file)[0]

        print(f"The image {img_full_path} will be read in.")
        img = Image.open(img_full_path).convert("L")

        img = np.array(img)
        h, w = img.shape


        img_grad = cv2.Laplacian(img, cv2.CV_8U)

        mask = np.zeros((h, w), dtype='int')
        mask[img_grad>args.threshold] = 1


        mask_255 = mask * 255
        mask_255 = np.clip(mask_255, 0, 255)
        mask_255 = mask_255.astype(np.uint8)
        mask_255 = np.ascontiguousarray(mask_255)
        mask_255 = Image.fromarray(mask_255)

        mask_255.save(os.path.join(save_path_png, f"{img_name}.png"), "PNG", quality = 95)
        io.savemat(os.path.join(save_path_mat, f"{img_name}.mat"), {'mat':mask}, do_compression=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/data0/chendu/dataset/DIV2K/DIV2K_multiscaleHR_subimages512')
    parser.add_argument('--save', type=str, default='/data0/chendu/dataset/DIV2K/mask_selfsim/DIV2K_multiscaleHR_subimages512/threshold-20')
    parser.add_argument("--threshold", type=float, default = 20.0)
    args = parser.parse_args()
    main(args)