import argparse
import cv2
import glob
import numpy as np
import os
import torch
import math

from basicsr.archs.rrdbnet_arch import RRDBNet


def tile_process(lq, net, args):
    """Modified from: https://github.com/ata4/esrgan-launcher
    """
    scale = args.scale
    img = lq
    batch, channel, height, width = img.shape
    output_height = height * scale
    output_width = width * scale
    output_shape = (batch, channel, output_height, output_width)

    # start with black image
    output = img.new_zeros(output_shape)
    tiles_x = math.ceil(width / args.tile_size)
    tiles_y = math.ceil(height / args.tile_size)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * args.tile_size
            ofs_y = y * args.tile_size
            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + args.tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + args.tile_size, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - args.tile_pad, 0)
            input_end_x_pad = min(input_end_x + args.tile_pad, width)
            input_start_y_pad = max(input_start_y - args.tile_pad, 0)
            input_end_y_pad = min(input_end_y + args.tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            tile_idx = y * tiles_x + x + 1
            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # upscale tile
            try:
                with torch.no_grad():
                    output_tile = net(input_tile)
            except Exception as error:
                print('Error', error)
            print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image
            output[:, :, output_start_y:output_end_y,
            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                           output_start_x_tile:output_end_x_tile]
    return output



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        '/data1_ssd4t/chendu/myprojects/EFDM_SIM/experiments/RRDB_SelfSim_v17_t20_gtsize128_ft_2.01_Bicubic_x4/models/net_g_150000.pth'  # noqa: E501
    )
    parser.add_argument('--input', type=str, default='/home/chendu/data2_hdd10t/chendu/dataset/basicsr/Urban100/LRbicx4mod24', help='input test image folder')
    parser.add_argument('--output', type=str, default='/data1_ssd4t/chendu/myprojects/EFDM_SIM/results/New/visualization/Urban100_', help='output folder')
    parser.add_argument('--tile_size', type=int, default=800)
    parser.add_argument('--tile_pad', type=int, default=32)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--suffix', type=str, default='_LDLBLindDF2KOST')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
    model.load_state_dict(torch.load(args.model_path)['params_ema'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)
    if args.resume:
        idx_end = len(os.listdir(args.output))
        resume_file_name = sorted(os.listdir(args.output))
        # resume_img_name = os.path.splitext(resume_file_name)[0]
        input_path_new = sorted(glob.glob(os.path.join(args.input, '*')))[idx_end:]
        print(f'Resume from {resume_file_name}.')
        for idx, path in enumerate(input_path_new):
            imgname = os.path.splitext(os.path.basename(path))[0]
            print('Testing', idx + idx_end, imgname)
            # read image
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = img.unsqueeze(0).to(device)
            # inference
            try:
                with torch.no_grad():
                    # output = model(img)
                    output = tile_process(img, model, args)
            except Exception as error:
                print('Error', error, imgname)
            else:
                # save image
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                output = (output * 255.0).round().astype(np.uint8)
                cv2.imwrite(os.path.join(args.output, f'{imgname}{args.suffix}.png'), output)
    else:
        for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
            imgname = os.path.splitext(os.path.basename(path))[0]
            print('Testing', idx, imgname)
            # read image
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = img.unsqueeze(0).to(device)
            # inference
            try:
                with torch.no_grad():
                    # output = model(img)
                    output = tile_process(img, model, args)
            except Exception as error:
                print('Error', error, imgname)
            else:
                # save image
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                output = (output * 255.0).round().astype(np.uint8)
                cv2.imwrite(os.path.join(args.output, f'{imgname}{args.suffix}.png'), output)


if __name__ == '__main__':
    main()
