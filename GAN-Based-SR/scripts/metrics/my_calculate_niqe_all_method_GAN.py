import argparse
import cv2
import os
import warnings

from basicsr.metrics import calculate_niqe
from basicsr.utils import scandir


def main(args):
    for idx, mtd in enumerate(args.input):
        methods = os.listdir(mtd)
        for method in methods:
            dataset_path_all_one_method = os.path.join(mtd, method, 'visualization')
            dataset_paths = os.listdir(dataset_path_all_one_method)
            for dataset in dataset_paths:

                dataset_path = os.path.join(dataset_path_all_one_method, dataset)

                save_txt_path = os.path.join(os.path.dirname(os.path.dirname(dataset_path)), f"NIQE_{dataset}.txt")
                save_txt = open(save_txt_path, mode='w', encoding='utf-8')

                niqe_all = []
                img_list = sorted(scandir(dataset_path, recursive=True, full_path=True))

                print(f"----------------------------")
                print(f"Now the testing dataset is {dataset_path}")
                for i, img_path in enumerate(img_list):
                    basename, _ = os.path.splitext(os.path.basename(img_path))
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        niqe_score = calculate_niqe(img, args.crop_border, input_order='HWC', convert_to='y')
                    print(f'{i+1:3d}: {basename:25}. \tNIQE: {niqe_score:.6f}')
                    save_txt.write(f"{basename}. \tNIQE: {niqe_score:.6f}\n")

                    niqe_all.append(niqe_score)


                print(f"Average NIQE for {dataset}: {sum(niqe_all) / len(niqe_all):.6f}")
                print(f"----------------------------")
                save_txt.write(f"Average NIQE for {dataset}: {sum(niqe_all) / len(niqe_all):.6f}")

                save_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs = '+', default=['/home/chendu/data2_hdd10t/chendu/myprojects/ISSL_Results/GAN'], help='Input path')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    args = parser.parse_args()
    main(args)
