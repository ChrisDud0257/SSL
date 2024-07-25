import argparse
import glob
import os
from PIL import Image


def main(args):
    scale_list = [0.75, 0.6, 1 / 3]
    shortest_edge = args.shortest_edge

    dataset = args.input
    path_list = sorted(glob.glob(os.path.join(dataset, '*')))
    save_path = args.save
    os.makedirs(save_path, exist_ok=True)
    for path in path_list:
        print(path)
        basename = os.path.splitext(os.path.basename(path))[0]

        img = Image.open(path)
        width, height = img.size
        width_original, height_original = img.size
        for idx, scale in enumerate(scale_list):
            # print(f'\t{scale:.2f}')

            if min(int(width * scale), int(height * scale)) >= shortest_edge:

                rlt = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
                rlt.save(os.path.join(save_path, f'{basename}T{idx}.png'))

        # save the smallest image which the shortest edge is 512
        if width_original < height_original :
            ratio = height_original / width_original
            width = shortest_edge
            height = int(width * ratio)
        else:
            ratio = width_original / height_original
            height = shortest_edge
            width = int(height * ratio)

        assert min(height, width) >= shortest_edge, f" The width-height {width}-{height} is not suitable."

        rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        rlt.save(os.path.join(save_path, f'{basename}S.png'))

        if min(width_original, height_original) >= shortest_edge:
            os.system(f'cp -r {path} {save_path}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/data0/chendu/dataset/DIV2K/GT', help='Input folder')
    parser.add_argument('--save', type = str, default = '/data0/chendu/dataset/DIV2K/DIV2K_multiscaleHR')
    parser.add_argument('--shortest_edge', type = int, default=512)
    args = parser.parse_args()
    main(args)
