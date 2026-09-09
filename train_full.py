"""Controlled full-label baseline and optional output refinement for ISIC."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=['vmunet', 'mambalite'], default='vmunet')
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--eta-min', type=float, default=0.00001)
    parser.add_argument('--t-max', type=int, default=None, help='Default: epochs; 50 audits old LR schedule')
    parser.add_argument('--preprocessing', choices=['corrected', 'legacy'], default='corrected')
    parser.add_argument('--boundary-weight', type=float, default=0.0, help='0 = baseline; >0 = edge BCE probe')
    parser.add_argument('--output-refine', action='store_true', help='Enable the 3x3 residual output refinement (R1)')
    parser.add_argument('--decoder', choices=['original', 'multiscale', 'context'], default='original',
                        help='context: preserve Mamba decoder with multiscale residual context (R3); multiscale: R2')
    parser.add_argument('--resume', action='store_true', help='Resume latest.pth in the same output directory')
    parser.add_argument('--init-checkpoint', help='Controlled B0 checkpoint for a new continuation experiment')
    parser.add_argument('--teacher-weight', type=float, default=0.0, help='Correct/confident training-pixel KL weight; 0 disables teacher')
    parser.add_argument('--teacher-confidence', type=float, default=0.9)
    args = parser.parse_args()
    if args.model == 'mambalite':
        if (args.pretrained or args.init_checkpoint or args.teacher_weight or
                args.decoder != 'original' or args.output_refine or args.boundary_weight):
            parser.error('MambaLite comparison requires random initialization and no VM-UNet variants/teacher')
    elif args.pretrained is None:
        args.pretrained = './pre_trained_weights/vmamba_small_e238_ema.pth'

    if not 0 <= args.teacher_weight < float('inf') or not 0.5 < args.teacher_confidence < 1:
        parser.error('Invalid teacher weight/confidence')
    if args.teacher_weight and not args.init_checkpoint:
        parser.error('--teacher-weight requires --init-checkpoint')
    if args.init_checkpoint and (args.decoder == 'multiscale' or args.output_refine or args.boundary_weight):
        parser.error('B0 continuation supports original/context with the original supervised loss')
    if args.decoder != 'original' and args.output_refine:
        parser.error('Custom decoders and --output-refine are separate model variants')
    args.t_max = args.t_max or args.epochs
    if min(args.epochs, args.batch_size, args.size, args.t_max) < 1 or args.size % 32:
        parser.error('Positive epochs/batch/t-max and size divisible by 32 required')
    if args.boundary_weight < 0 or args.lr <= 0 or not 0 <= args.eta_min <= args.lr or args.weight_decay < 0:
        parser.error('Invalid loss or optimizer settings')
    return args


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import random
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from fullsup.data import load_manifest
    from fullsup.runtime import (SegmentationDataset, atomic_save, build_model, seed_everything,
                                 segmentation_loss, validate, write_json, initialize_from_baseline,
                                 teacher_constraint)
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires the existing CUDA/Mamba environment')
    manifest = load_manifest(args.data_path, args.manifest)
    output = Path(args.output)
    config = vars(args).copy()
    config.pop('resume')
    if args.model == 'mambalite':
        config['model_sha256'] = hashlib.sha256(
            (Path(__file__).parent / 'models/mambalite.py').read_bytes()).hexdigest()
    config['manifest_sha256'] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if args.init_checkpoint:
        digest = hashlib.sha256()
        with open(args.init_checkpoint, 'rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        config['init_sha256'] = digest.hexdigest()
    if args.resume:
        saved_config = json.loads((output / 'config.json').read_text())
        saved_config.setdefault('model', 'vmunet')
        saved_config.setdefault('output_refine', False)
        saved_config.setdefault('decoder', 'original')
        saved_config.setdefault('init_checkpoint', None)
        saved_config.setdefault('teacher_weight', 0.0)
        saved_config.setdefault('teacher_confidence', 0.9)
        if saved_config != config:
            raise ValueError('Resume config differs; use original arguments or a fresh output directory')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'config.json', config)
        write_json(output / 'manifest.json', manifest)
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        import torchvision
        write_json(output / 'environment.json', dict(torch=torch.__version__, torchvision=torchvision.__version__,
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                   git_commit=revision.stdout.strip(), git_status=dirty.stdout.strip(),
                   selection='maximum validation pooled_dice, threshold=0.5',
                   note='All training masks used. Validation is not an independent test.'))
    seed_everything(args.seed)
    train_ds = SegmentationDataset(args.data_path, manifest['train'], args.size, True, args.preprocessing)
    val_ds = SegmentationDataset(args.data_path, manifest['val'], args.size, False, args.preprocessing)
    # workers=0 makes augmentation RNG checkpointable; no cached cycle(loader).
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    teacher = None
    if args.init_checkpoint:
        model, teacher = initialize_from_baseline(args.init_checkpoint, config, bool(args.teacher_weight))
        if teacher is not None:
            teacher = teacher.cuda()
    else:
        model = build_model(None if args.resume else args.pretrained,
                            output_refine=args.output_refine, decoder=args.decoder, model_name=args.model)
    model = model.cuda()
    print('Model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)
    start, best = 1, -1.0
    history = []
    if args.resume:
        state = torch.load(output / 'latest.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start, best = state['epoch'] + 1, state['best_dice']
        history = state['history']
        random.setstate(state['rng_python'])
        np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch'])
        torch.cuda.set_rng_state_all(state['rng_cuda'])
        generator.set_state(state['rng_loader'])
    elif args.init_checkpoint:
        # Both the continuation control and proposed method retain their starting best.
        metrics = validate(model, val_loader, 'cuda')
        best = metrics['pooled_dice']
        atomic_save(dict(model_state_dict=model.state_dict(), config=config, epoch=0,
                         metrics=metrics), output / 'best.pth')
        write_json(output / 'initial_metrics.json', metrics)
    for epoch in range(start, args.epochs + 1):
        started = time.monotonic()
        model.train()
        if args.model == 'mambalite':
            torch.cuda.reset_peak_memory_stats()
        train_loss = 0.0
        teacher_loss_sum = teacher_coverage_sum = 0.0
        lr = optimizer.param_groups[0]['lr']
        for image, target, _ in train_loader:
            image, target = image.cuda(), target.cuda()
            optimizer.zero_grad(set_to_none=True)
            prob = model(image)
            if not torch.isfinite(prob).all():
                raise FloatingPointError('Non-finite training output at epoch {}'.format(epoch))
            loss = segmentation_loss(prob, target, args.boundary_weight)
            if teacher is not None:
                with torch.no_grad():
                    teacher_prob = teacher(image)
                regularizer, coverage = teacher_constraint(prob, teacher_prob, target, args.teacher_confidence)
                loss = loss + args.teacher_weight * regularizer
                teacher_loss_sum += regularizer.item() * len(image)
                teacher_coverage_sum += coverage.item() * len(image)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss at epoch {}'.format(epoch))
            loss.backward()
            # Detect invalid gradients without clipping/changing the baseline optimization.
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            train_loss += loss.item() * len(image)
        metrics = validate(model, val_loader, 'cuda')
        scheduler.step()
        row = dict(epoch=epoch, lr=lr, train_loss=train_loss / len(train_ds),
                   **metrics, seconds=time.monotonic() - started)
        if args.model == 'mambalite':
            row['peak_cuda_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        if teacher is not None:
            row.update(teacher_kl=teacher_loss_sum / len(train_ds),
                       teacher_coverage=teacher_coverage_sum / len(train_ds))
        history.append(row)
        if metrics['pooled_dice'] > best:
            best = metrics['pooled_dice']
            atomic_save(dict(model_state_dict=model.state_dict(), config=config, epoch=epoch,
                             metrics=metrics), output / 'best.pth')
        atomic_save(dict(epoch=epoch, best_dice=best, model_state_dict=model.state_dict(),
                         optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                         config=config, history=history, rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                         rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                         rng_loader=generator.get_state()), output / 'latest.pth')
        with (output / 'metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)
    print('Saved through epoch {}/{}. Best validation pooled Dice: {:.6f}. No test set evaluated.'.format(
        history[-1]['epoch'] if history else 0, args.epochs, best))


if __name__ == '__main__':
    main()
