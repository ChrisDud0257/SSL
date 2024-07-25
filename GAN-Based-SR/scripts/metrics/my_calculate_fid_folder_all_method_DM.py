import argparse
import math
import os.path

import numpy as np
import torch
from torch.utils.data import DataLoader

from basicsr.data import build_dataset
from basicsr.metrics.fid import calculate_fid, extract_inception_features, load_patched_inception_v3


def calculate_fid_folder(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # inception model
    inception = load_patched_inception_v3(device)

    for idx, mtd in enumerate(args.folder):
        methods = os.listdir(mtd)
        for method in methods:
            dataset_path_all_one_method = os.path.join(mtd, method, 'visualization')
            dataset_paths = os.listdir(dataset_path_all_one_method)

            if 'DPEDiphoneValSet_crop128' in dataset_paths:
                dataset_paths.remove('DPEDiphoneValSet_crop128')

            for dataset in dataset_paths:
                dataset_path = os.path.join(dataset_path_all_one_method, dataset)
                fid_stats_path = os.path.join(args.fid_stats[idx], dataset, 'FID_status', f'{dataset}_512.pth')
                print(f"----------------------------")
                print(f"fid_stats_path is {fid_stats_path}")
                print(f"Now the testing dataset is {dataset_path}")

                opt = {}
                opt['name'] = dataset
                opt['type'] = 'SingleImageDataset'
                opt['dataroot_lq'] = dataset_path
                opt['io_backend'] = dict(type=args.backend)
                opt['mean'] = [0.5, 0.5, 0.5]
                opt['std'] = [0.5, 0.5, 0.5]
                dataset = build_dataset(opt)

                print(f"length dataset is {len(dataset)}")

                save_txt_path = os.path.join(os.path.dirname(os.path.dirname(dataset_path)), f"FID_{opt['name']}.txt")
                save_txt = open(save_txt_path, mode='w', encoding='utf-8')

                # create dataloader
                data_loader = DataLoader(
                    dataset=dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    sampler=None,
                    drop_last=False)

                total_batch = math.ceil(len(dataset) / args.batch_size)

                print(f"total_batch is {total_batch}")

                def data_generator(data_loader, total_batch):
                    for idx, data in enumerate(data_loader):
                        if idx >= total_batch:
                            break
                        else:
                            yield data['lq']

                features = extract_inception_features(data_generator(data_loader, total_batch), inception, total_batch, device)
                features = features.numpy()
                total_len = features.shape[0]
                print(f"features shape is {features.shape}")
                features = features[:args.num_sample]
                print(f'Extracted {total_len} features, use the first {features.shape[0]} features to calculate stats.')

                sample_mean = np.mean(features, 0)
                sample_cov = np.cov(features, rowvar=False)

                # load the dataset stats
                stats = torch.load(fid_stats_path)
                real_mean = stats['mean']
                real_cov = stats['cov']

                # calculate FID metric
                fid = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)
                print(f"FID for {opt['name']}: {fid}")
                save_txt.write(f"FID for {opt['name']}: {fid:.6f}")
                print(f"----------------------------")

                save_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', nargs='+', default=['/home/chendu/data2_hdd10t/chendu/myprojects/ISSL_Results/ResShift-SSL-AblationStudy'])
    # parser.add_argument('--fid_stats', nargs='+', default=['/data0/chendu/dataset/StableSR-GT-LQ'])
    parser.add_argument('--fid_stats', nargs='+', default=['/home/chendu/data2_hdd10t/chendu/dataset/StableSR-GT-LQ'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_sample', type=int, default=50000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--backend', type=str, default='disk', help='io backend for dataset. Option: disk, lmdb')
    args = parser.parse_args()
    calculate_fid_folder(args)
