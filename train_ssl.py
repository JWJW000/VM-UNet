import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from configs.config_ssl import setting_config_ssl
from datasets.dataset import NPY_datasets
from engine import test_one_epoch, val_one_epoch
from models.vmunet.vmunet import VMUNet
from scan_ssl.engine import build_teacher, train_one_epoch_ssl
from scan_ssl.splits import make_labeled_split
from utils import cal_params_flops, get_logger, get_optimizer, get_scheduler, log_config_info, set_seed

import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Scan-aware semi-supervised VM-UNet')
    parser.add_argument('--labeled_ratio', type=float, default=None)
    parser.add_argument('--datasets', type=str, default=None, choices=['isic17', 'isic18'])
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no_dir_consistency', action='store_true')
    parser.add_argument('--no_scan_dropout', action='store_true')
    return parser.parse_args()


def apply_args(config, args):
    if args.labeled_ratio is not None:
        config.labeled_ratio = args.labeled_ratio
    if args.datasets is not None:
        config.datasets = args.datasets
        if args.datasets == 'isic18':
            config.data_path = './data/isic2018/'
        elif args.datasets == 'isic17':
            config.data_path = './data/isic2017/'
    if args.data_path is not None:
        config.data_path = args.data_path
    if args.gpu is not None:
        config.gpu_id = args.gpu
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.seed is not None:
        config.ssl_seed = args.seed
        config.seed = args.seed
    if args.no_dir_consistency:
        config.use_dir_consistency = False
    if args.no_scan_dropout:
        config.use_scan_dropout = False

    ratio_tag = str(config.labeled_ratio).replace('.', 'p')
    from datetime import datetime
    config.work_dir = (
        'results/'
        + config.network
        + '_ssl_'
        + config.datasets
        + f'_r{ratio_tag}_'
        + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
        + '/'
    )
    return config


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    logger = get_logger('train_ssl', log_dir)
    writer = SummaryWriter(config.work_dir + 'summary')
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing SSL split----------#')
    split, split_file = make_labeled_split(
        config.data_path,
        config.datasets,
        config.labeled_ratio,
        config.ssl_seed,
        split_dir=config.split_dir,
    )
    logger.info(f'split file: {split_file}')
    logger.info(
        f"n_total={split['n_total']} n_labeled={split['n_labeled']} n_unlabeled={split['n_unlabeled']}"
    )
    if split['n_unlabeled'] == 0:
        raise RuntimeError('unlabeled set is empty; raise labeled_ratio or check the dataset')

    labeled_dataset = NPY_datasets(config.data_path, config, train=True, file_list=split['labeled'])
    unlabeled_dataset = NPY_datasets(config.data_path, config, train=True, file_list=split['unlabeled'])
    val_dataset = NPY_datasets(config.data_path, config, train=False)

    labeled_bs = min(config.labeled_batch_size, max(1, len(labeled_dataset)))
    unlabeled_bs = min(config.unlabeled_batch_size, max(1, len(unlabeled_dataset)))
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=labeled_bs,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=len(labeled_dataset) >= labeled_bs,
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=unlabeled_bs,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=len(unlabeled_dataset) >= unlabeled_bs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True,
    )

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    student = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    student.load_from()
    student = student.cuda()
    teacher = build_teacher(student)
    cal_params_flops(student, 256, logger)

    criterion = config.criterion
    optimizer = get_optimizer(config, student)
    scheduler = get_scheduler(config, optimizer)

    min_loss = 999
    min_epoch = 1
    start_epoch = 1
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        student.load_state_dict(checkpoint['model_state_dict'])
        teacher.load_state_dict(checkpoint['teacher_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch = checkpoint['min_loss'], checkpoint['min_epoch']
        logger.info(f'resuming ssl from {resume_model}, epoch {saved_epoch}')

    step = 0
    print('#----------SSL Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        step = train_one_epoch_ssl(
            labeled_loader,
            unlabeled_loader,
            student,
            teacher,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
        )
        loss = val_one_epoch(val_loader, student, criterion, epoch, logger, config)
        if loss < min_loss:
            torch.save(student.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch
        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'split_file': split_file,
            },
            os.path.join(checkpoint_dir, 'latest.pth'),
        )

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location='cpu')
        student.load_state_dict(best_weight)
        test_one_epoch(val_loader, student, criterion, logger, config)
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth'),
        )


if __name__ == '__main__':
    args = parse_args()
    config = apply_args(setting_config_ssl, args)
    main(config)
