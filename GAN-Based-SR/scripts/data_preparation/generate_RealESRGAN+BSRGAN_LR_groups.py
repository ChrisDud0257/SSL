import os
import argparse
import cv2
from PIL import Image
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels, random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils.img_process_util import filter2D
from basicsr.utils import DiffJPEG, USMSharp
import yaml
from collections import OrderedDict
import numpy as np
import random
import math
import torch
import torch.nn.functional as F

from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, imfromfile
import train_BSGRAN.utils.utils_image as util
from train_BSGRAN.utils import utils_parameter_blindsr as blindsr

# def set_random_seed(seed):
#     """Set random seeds."""
#     random.seed(seed)
#     np.random.seed(seed)

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def main(args):
    with open(args.param_path, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    # set_random_seed(opt['manual_seed'])
    jpeger = DiffJPEG(differentiable=False)
    usm_sharpener = USMSharp()

    for dataset, dataroot in opt['datasets'].items():
        print(f'dataset is {dataset}.')
        print(f"dataroot is {dataroot['dataroot']}.")
        save_LR_full_path = os.path.join(args.save_LR_path, os.path.splitext(os.path.basename(args.param_path))[0]+f"-groups{args.groups}",
                                         dataset)
        save_bicubicSR_full_path = os.path.join(args.save_bicubicSR_path,
                                                os.path.splitext(os.path.basename(args.param_path))[0] + f'-bicubicSR-groups{args.groups}',
                                                dataset)
        os.makedirs(save_LR_full_path, exist_ok=True)
        os.makedirs(save_bicubicSR_full_path, exist_ok=True)
        for idx in range(args.groups):
            print(f"Groups is {idx + 1}")

            for file in os.listdir(dataroot['dataroot']):
                if random.random() > 0.5:
                    print(f"RealESRGAN degradation")
                    img_name, ext = os.path.splitext(file)
                    img = imfromfile(os.path.join(dataroot['dataroot'], file), float32=True)
                    opt_RealESRGAN = opt['RealESRGAN']

                    # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
                    # blur settings for the first degradation
                    blur_kernel_size_min = opt_RealESRGAN['blur_kernel_size_min']
                    blur_kernel_size_max = opt_RealESRGAN['blur_kernel_size_max']
                    kernel_list = opt_RealESRGAN['kernel_list']
                    kernel_prob = opt_RealESRGAN['kernel_prob']  # a list for each kernel probability
                    blur_sigma = opt_RealESRGAN['blur_sigma']
                    betag_range = opt_RealESRGAN['betag_range']  # betag used in generalized Gaussian blur kernels
                    betap_range = opt_RealESRGAN['betap_range']  # betap used in plateau blur kernels
                    sinc_prob = opt_RealESRGAN['sinc_prob']  # the probability for sinc filters
                    kernel_range = [2 * v + 1 for v in range(blur_kernel_size_min, blur_kernel_size_max + 1)]  # kernel size ranges from 3 to 9
                    kernel_size = random.choice(kernel_range)

                    if np.random.uniform() < sinc_prob:
                        # this sinc filter setting is for kernels ranging from [7, 21]
                        if kernel_size < 13:
                            omega_c = np.random.uniform(np.pi / 3, np.pi)
                        else:
                            omega_c = np.random.uniform(np.pi / 5, np.pi)
                        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
                    else:
                        kernel = random_mixed_kernels(
                            kernel_list,
                            kernel_prob,
                            kernel_size,
                            blur_sigma,
                            blur_sigma, [-math.pi, math.pi],
                            betag_range,
                            betap_range,
                            noise_range=None)
                    # pad kernel
                    pad_size = (kernel_range[-1] - kernel_size) // 2
                    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

                    img = img2tensor([img], bgr2rgb=True, float32=True)[0]
                    img = img.unsqueeze(0)
                    kernel = torch.FloatTensor(kernel)

                    ori_h, ori_w = img.size()[2:4]

                    if opt_RealESRGAN.get('Use_sharpen', None) is not None:
                        gt_usm = usm_sharpener(img)

                    # ----------------------- The first degradation process ----------------------- #
                    # blur
                    if opt_RealESRGAN.get('Use_sharpen', None) is not None and opt_RealESRGAN.get('Sharpen_before_degra', False):
                        out = filter2D(gt_usm, kernel)
                    else:
                        out = filter2D(img, kernel)
                    # random resize
                    updown_type = random.choices(['up', 'down', 'keep'], opt_RealESRGAN['resize_prob'])[0]
                    if updown_type == 'up':
                        scale = np.random.uniform(1, opt_RealESRGAN['resize_range'][1])
                    elif updown_type == 'down':
                        scale = np.random.uniform(opt_RealESRGAN['resize_range'][0], 1)
                    else:
                        scale = 1
                    mode = random.choice(['area', 'bilinear', 'bicubic'])
                    out = F.interpolate(out, scale_factor=scale, mode=mode)

                    # add noise
                    gray_noise_prob = opt_RealESRGAN['gray_noise_prob']
                    if np.random.uniform() < opt_RealESRGAN['gaussian_noise_prob']:
                        out = random_add_gaussian_noise_pt(
                            out, sigma_range=opt_RealESRGAN['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                    else:
                        out = random_add_poisson_noise_pt(
                            out,
                            scale_range=opt_RealESRGAN['poisson_scale_range'],
                            gray_prob=gray_noise_prob,
                            clip=True,
                            rounds=False)
                    # JPEG compression
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt_RealESRGAN['jpeg_range'])
                    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
                    out = jpeger(out, quality=jpeg_p)

                    if random.random() < opt_RealESRGAN['use_second_order_prob']:
                        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
                        # blur settings for the second degradation
                        second_blur_prob = opt_RealESRGAN['second_blur_prob']
                        blur_kernel2_size_min = opt_RealESRGAN['blur_kernel_size_min2']
                        blur_kernel2_size_max = opt_RealESRGAN['blur_kernel_size_max2']
                        kernel_list2 = opt_RealESRGAN['kernel_list2']
                        kernel_prob2 = opt_RealESRGAN['kernel_prob2']
                        blur_sigma2 = opt_RealESRGAN['blur_sigma2']
                        betag_range2 = opt_RealESRGAN['betag_range2']
                        betap_range2 = opt_RealESRGAN['betap_range2']
                        sinc_prob2 = opt_RealESRGAN['sinc_prob2']

                        # a final sinc filter
                        final_sinc_prob = opt_RealESRGAN['final_sinc_prob']

                        kernel_range = [2 * v + 1 for v in range(blur_kernel2_size_min, blur_kernel2_size_max + 1)]  # kernel size ranges from 7 to 21
                        pulse_tensor = torch.zeros(kernel_range[-1], kernel_range[-1]).float()  # convolving with pulse tensor brings no blurry effect
                        pulse_tensor[kernel_range[-1]//2, kernel_range[-1]//2] = 1
                        kernel_size = random.choice(kernel_range)
                        if np.random.uniform() < sinc_prob2:
                            if kernel_size < 13:
                                omega_c = np.random.uniform(np.pi / 3, np.pi)
                            else:
                                omega_c = np.random.uniform(np.pi / 5, np.pi)
                            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
                        else:
                            kernel2 = random_mixed_kernels(
                                kernel_list2,
                                kernel_prob2,
                                kernel_size,
                                blur_sigma2,
                                blur_sigma2, [-math.pi, math.pi],
                                betag_range2,
                                betap_range2,
                                noise_range=None)

                        # pad kernel
                        pad_size = (kernel_range[-1] - kernel_size) // 2
                        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

                        # ------------------------------------- the final sinc kernel ------------------------------------- #
                        if np.random.uniform() < final_sinc_prob:
                            kernel_size = random.choice(kernel_range)
                            omega_c = np.random.uniform(np.pi / 3, np.pi)
                            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                            sinc_kernel = torch.FloatTensor(sinc_kernel)
                        else:
                            sinc_kernel = pulse_tensor

                        kernel2 = torch.FloatTensor(kernel2)

                        # ----------------------- The second degradation process ----------------------- #
                        # blur
                        if np.random.uniform() < second_blur_prob:
                            out = filter2D(out, kernel2)
                        # random resize
                        updown_type = random.choices(['up', 'down', 'keep'], opt_RealESRGAN['resize_prob2'])[0]
                        if updown_type == 'up':
                            scale = np.random.uniform(1, opt_RealESRGAN['resize_range2'][1])
                        elif updown_type == 'down':
                            scale = np.random.uniform(opt_RealESRGAN['resize_range2'][0], 1)
                        else:
                            scale = 1
                        mode = random.choice(['area', 'bilinear', 'bicubic'])
                        out = F.interpolate(
                            out, size=(int(ori_h / opt_RealESRGAN['scale'] * scale), int(ori_w / opt_RealESRGAN['scale'] * scale)),
                            mode=mode)
                        # add noise
                        gray_noise_prob = opt_RealESRGAN['gray_noise_prob2']
                        if np.random.uniform() < opt_RealESRGAN['gaussian_noise_prob2']:
                            out = random_add_gaussian_noise_pt(
                                out, sigma_range=opt_RealESRGAN['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                        else:
                            out = random_add_poisson_noise_pt(
                                out,
                                scale_range=opt_RealESRGAN['poisson_scale_range2'],
                                gray_prob=gray_noise_prob,
                                clip=True,
                                rounds=False)

                        # JPEG compression + the final sinc filter
                        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
                        # as one operation.
                        # We consider two orders:
                        #   1. [resize back + sinc filter] + JPEG compression
                        #   2. JPEG compression + [resize back + sinc filter]
                        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
                        if np.random.uniform() < 0.5:
                            # resize back + the final sinc filter
                            mode = random.choice(['area', 'bilinear', 'bicubic'])
                            out = F.interpolate(out, size=(ori_h // opt_RealESRGAN['scale'], ori_w // opt_RealESRGAN['scale']), mode=mode)
                            out = filter2D(out, sinc_kernel)
                            # JPEG compression
                            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt_RealESRGAN['jpeg_range2'])
                            out = torch.clamp(out, 0, 1)
                            out = jpeger(out, quality=jpeg_p)
                        else:
                            # JPEG compression
                            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt_RealESRGAN['jpeg_range2'])
                            out = torch.clamp(out, 0, 1)
                            out = jpeger(out, quality=jpeg_p)
                            # resize back + the final sinc filter
                            mode = random.choice(['area', 'bilinear', 'bicubic'])
                            out = F.interpolate(out, size=(ori_h // opt_RealESRGAN['scale'], ori_w // opt_RealESRGAN['scale']), mode=mode)
                            out = filter2D(out, sinc_kernel)
                    else:
                        out = F.interpolate(out, size=(ori_h // opt_RealESRGAN['scale'], ori_w // opt_RealESRGAN['scale']), mode=mode)

                    lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
                    lq = lq.data.squeeze().float().clamp_(0, 1).numpy()
                    lq = np.transpose(lq[[2, 1, 0], :, :], (1, 2, 0))
                    lq = (lq * 255.0).round().astype(np.uint8)

                    bicubicSR = F.interpolate(out, size = (ori_h, ori_w), mode='bicubic')
                    bicubicSR = torch.clamp((bicubicSR * 255.0).round(), 0, 255) / 255.
                    bicubicSR = bicubicSR.data.squeeze().float().clamp_(0, 1).numpy()
                    bicubicSR = np.transpose(bicubicSR[[2, 1, 0], :, :], (1, 2, 0))
                    bicubicSR = (bicubicSR * 255.0).round().astype(np.uint8)

                    cv2.imwrite(os.path.join(save_LR_full_path, f'{img_name}+{idx}.png'), lq)
                    cv2.imwrite(os.path.join(save_bicubicSR_full_path, f'{img_name}+{idx}_bicubic.png'), bicubicSR)
                else:
                    print(f"BSRGAN degradation")
                    img_name, ext = os.path.splitext(file)
                    img_path = os.path.join(dataroot['dataroot'], file)
                    img_H = util.imread_uint(img_path)
                    h, w, c = img_H.shape
                    img_H = util.uint2single(img_H)

                    opt_BSRGAN = opt['BSRGAN']
                    lq, _ = blindsr.degradation_bsrgan_nocrop(img_H, opt=opt_BSRGAN, sf=4, lq_patchsize=16, isp_model=None)

                    bicubicSR = cv2.resize(lq, (w, h), interpolation=cv2.INTER_CUBIC)

                    lq = cv2.cvtColor(lq, cv2.COLOR_RGB2BGR)
                    bicubicSR = cv2.cvtColor(bicubicSR, cv2.COLOR_RGB2BGR)

                    lq = np.clip((lq * 255.0).round(), 0, 255).astype(np.uint8)
                    bicubicSR = np.clip((bicubicSR * 255.0).round(), 0, 255).astype(np.uint8)

                    cv2.imwrite(os.path.join(save_LR_full_path, f'{img_name}+{idx}.png'), lq)
                    cv2.imwrite(os.path.join(save_bicubicSR_full_path, f'{img_name}+{idx}_bicubic.png'), bicubicSR)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--param_path', type=str, default='/data1_ssd4t/chendu/myprojects/EFDM_SIM/options/generate_blind_LR/test_RealESRGAN+BSRGAN_LR.yml')
    parse.add_argument('--save_LR_path', type=str, default='/home/chendu/data2_hdd10t/chendu/dataset/BlindLR/ISSL/LR_x4')
    parse.add_argument('--save_bicubicSR_path', type=str, default='/home/chendu/data2_hdd10t/chendu/dataset/BlindLR/ISSL/BicubicSR')
    parse.add_argument("--groups", type = int, default=10)
    args =  parse.parse_args()
    main(args)