import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

from basicsr.data import build_dataset
from basicsr.metrics.fid import extract_inception_features, load_patched_inception_v3


def calculate_stats_from_dataset(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # inception model
    inception = load_patched_inception_v3(device)

    for dataset in args.dataroot:
        print(dataset)
        save_path = os.path.join(os.path.dirname(dataset), 'FID_status')
        os.makedirs(save_path, exist_ok=True)
        # create dataset
        opt = {}
        opt['name'] = os.path.basename(os.path.dirname(dataset))
        opt['type'] = args.dataset_type
        opt['dataroot_lq'] = dataset
        opt['io_backend'] = dict(type='disk')
        opt['use_hflip'] = False
        opt['mean'] = [0.5, 0.5, 0.5]
        opt['std'] = [0.5, 0.5, 0.5]
        dataset = build_dataset(opt)

        # create dataloader
        data_loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, sampler=None, drop_last=False)
        total_batch = math.ceil(args.num_sample / args.batch_size)

        def data_generator(data_loader, total_batch):
            for idx, data in enumerate(data_loader):
                if idx >= total_batch:
                    break
                else:
                    yield data['lq']

        features = extract_inception_features(data_generator(data_loader, total_batch), inception, len_generator=total_batch, device = device)
        features = features.numpy()
        total_len = features.shape[0]
        features = features[:args.num_sample]
        print(f'Extracted {total_len} features, use the first {features.shape[0]} features to calculate stats.')
        mean = np.mean(features, 0)
        cov = np.cov(features, rowvar=False)

        save_path_fid_status = os.path.join(save_path, f'{opt["name"]}_{args.size}{args.ext}.pth')
        torch.save(
            dict(name=opt['name'], size=args.size, mean=mean, cov=cov), save_path_fid_status, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sample', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--dataroot', nargs='+', default=['/home/chendu/data2_hdd10t/chendu/dataset/basicsr/DIV2K100/GT/GTmod12'])
    parser.add_argument('--dataset_type', type = str, default='SingleImageDataset')
    parser.add_argument('--ext', type = str, default='')
    args = parser.parse_args()
    calculate_stats_from_dataset(args)
