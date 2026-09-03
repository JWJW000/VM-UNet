"""Cross-domain test: load a source-trained ckpt, evaluate on a target val set.

Normalization uses the *source* dataset statistics (no target peeking).
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from torchvision import transforms

from configs.config_setting import setting_config
from datasets.dataset import NPY_datasets
from engine import test_one_epoch
from models.vmunet.vmunet import VMUNet
from utils import get_logger, myNormalize, myResize, myToTensor, set_seed


DATASET_PATHS = {
    'isic17': './data/isic2017/',
    'isic18': './data/isic2018/',
}


def parse_args():
    parser = argparse.ArgumentParser(description='VM-UNet cross-domain evaluation')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--source', type=str, required=True, choices=['isic17', 'isic18'])
    parser.add_argument('--target', type=str, required=True, choices=['isic17', 'isic18'])
    parser.add_argument('--target_data_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--work_dir', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = setting_config
    config.gpu_id = args.gpu
    config.datasets = args.source
    target_path = args.target_data_path or DATASET_PATHS[args.target]
    work_dir = args.work_dir or (
        f'results/cross_domain_{args.source}_to_{args.target}/'
    )
    os.makedirs(work_dir + 'outputs/', exist_ok=True)
    os.makedirs(work_dir + 'log/', exist_ok=True)
    config.work_dir = work_dir

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(config.seed)

    logger = get_logger('cross_domain', work_dir + 'log')
    logger.info(f'source={args.source} target={args.target} ckpt={args.ckpt}')
    logger.info(f'target_data_path={target_path}')
    logger.info('normalize with source val statistics (no target fitting)')

    config.test_transformer = transforms.Compose([
        myNormalize(args.source, train=False),
        myToTensor(),
        myResize(config.input_size_h, config.input_size_w),
    ])

    val_dataset = NPY_datasets(target_path, config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True,
    )

    model_cfg = config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=None,
    )
    state = torch.load(args.ckpt, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model = model.cuda()

    test_one_epoch(
        val_loader,
        model,
        config.criterion,
        logger,
        config,
        test_data_name=f'{args.source}_to_{args.target}',
    )


if __name__ == '__main__':
    main()
