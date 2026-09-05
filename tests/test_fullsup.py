import json

import numpy as np
import pytest

from audit_training import parse_log
from fullsup.data import load_manifest, make_manifest, pairs_in
from fullsup.metrics import binary_metrics, summarize


def make_files(root, split, count):
    for folder in ('images', 'masks'):
        (root / split / folder).mkdir(parents=True)
    for index in range(count):
        (root / split / 'images' / '{}_{}.png'.format(split, index)).write_bytes(str(index).encode() + split.encode())
        (root / split / 'masks' / '{}_{}_segmentation.png'.format(split, index)).write_bytes(b'mask')


def test_pairing_and_internal_holdout(tmp_path):
    make_files(tmp_path, 'train', 10)
    make_files(tmp_path, 'val', 3)
    manifest = make_manifest(tmp_path, 0.2, 42)
    assert manifest == make_manifest(tmp_path, 0.2, 42)
    assert len(manifest['train']) == 8 and len(manifest['val']) == 2
    assert all(p[0].startswith('train/') for p in manifest['val'])
    path = tmp_path / 'split.json'
    path.write_text(json.dumps(manifest))
    assert load_manifest(tmp_path, path) == manifest
    (tmp_path / 'train/masks/train_0_segmentation.png').unlink()
    with pytest.raises(ValueError, match='match by stem'):
        pairs_in(tmp_path, 'train')


def test_reject_exact_copy_between_partitions(tmp_path):
    make_files(tmp_path, 'train', 3)
    make_files(tmp_path, 'val', 2)
    (tmp_path / 'val/images/val_0.png').write_bytes((tmp_path / 'train/images/train_0.png').read_bytes())
    with pytest.raises(ValueError, match='Identical image bytes'):
        make_manifest(tmp_path)


def test_overlap_boundary_and_empty_cases():
    a = np.zeros((20, 20), bool)
    a[5:15, 5:15] = True
    equal = binary_metrics(a, a)
    assert equal['dice'] == equal['boundary_f1'] == 1
    assert equal['hd95_px'] == 0
    shifted = np.roll(a, 1, axis=1)
    result = binary_metrics(shifted, a, tolerance=0)
    assert result['dice'] == pytest.approx(0.9)
    assert result['boundary_f1'] < 1
    assert result['hd95_px'] == 1
    empty = np.zeros_like(a)
    assert binary_metrics(empty, empty)['dice'] == 1
    failed = binary_metrics(empty, a)
    assert failed['dice'] == 0 and failed['hd95_px'] is None
    summary = summarize([equal, failed])
    assert summary['macro_dice'] == 0.5
    assert summary['pooled_dice'] == pytest.approx(2 / 3)
    assert summary['hd95_undefined_count'] == 1


def test_legacy_log_flags_drop_not_cause():
    rows, drops = parse_log('val epoch: 270, loss: 0.32, f1_or_dsc: 0.8889\n'
                           'val epoch: 271, loss: 0.31\n'
                           'val epoch: 300, loss: 0.98, f1_or_dsc: 0.5348\n')
    assert len(rows) == 3
    assert drops[0]['drop_percentage_points'] == pytest.approx(35.41)


def test_loss_and_validation_cpu(tmp_path):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from fullsup.runtime import atomic_save, segmentation_loss, validate
    logits = torch.randn(2, 1, 12, 12, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 3:9, 3:9] = 1
    p = logits.sigmoid()
    expected = torch.nn.functional.binary_cross_entropy(p, target) + 1 - (
        (2 * (p * target).flatten(1).sum(1) + 1) /
        (p.flatten(1).sum(1) + target.flatten(1).sum(1) + 1)).mean()
    assert torch.allclose(segmentation_loss(p, target, 0), expected)
    loss = segmentation_loss(p, target, 2)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    ds = TensorDataset(target, target, torch.arange(2))
    result = validate(torch.nn.Identity(), DataLoader(ds, batch_size=1), 'cpu')
    assert result['pooled_dice'] == result['macro_dice'] == 1
    atomic_save({'model_state_dict': {'x': target}}, tmp_path / 'best.pth')
    assert not (tmp_path / 'best.pth.tmp').exists()


def test_preprocessing_binary_masks_and_constant_images(tmp_path):
    import torch
    from PIL import Image
    from fullsup.runtime import SegmentationDataset
    image = np.zeros((13, 13, 3), np.uint8) + 120
    mask = np.zeros((13, 13), np.uint8)
    mask[3:9, 3:9] = 255
    Image.fromarray(image).save(tmp_path / 'image.png')
    Image.fromarray(mask).save(tmp_path / 'mask.png')
    ds = SegmentationDataset(tmp_path, [['image.png', 'mask.png']], 32)
    x, y, name = ds[0]
    assert torch.isfinite(x).all()
    assert set(y.unique().tolist()) == {0.0, 1.0}
    assert name == 'image.png'


def test_training_resume_matches_uninterrupted_cpu(tmp_path, monkeypatch):
    """Exercise the real runner loop with a tiny CPU model, including RNG restoration."""
    import sys
    import torch
    from PIL import Image
    import train_full
    import fullsup.runtime as runtime
    for split in ('train', 'val'):
        for folder in ('images', 'masks'):
            (tmp_path / split / folder).mkdir(parents=True)
        for index in range(2):
            rng = np.random.RandomState(index + (10 if split == 'val' else 0))
            image = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            mask = (image[:, :, 0] > 128).astype(np.uint8) * 255
            Image.fromarray(image).save(tmp_path / split / 'images' / '{}{}.png'.format(split, index))
            Image.fromarray(mask).save(tmp_path / split / 'masks' / '{}{}.png'.format(split, index))
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(make_manifest(tmp_path)))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'CPU test double')
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda _: None)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(runtime, 'build_model', lambda _: torch.nn.Sequential(
        torch.nn.Conv2d(3, 1, 1), torch.nn.Sigmoid()))
    original_validate = runtime.validate
    monkeypatch.setattr(runtime, 'validate', lambda model, loader, device: original_validate(model, loader, 'cpu'))
    # CPU DataLoader must not attempt to allocate CUDA-pinned memory.
    from torch.utils.data import DataLoader
    monkeypatch.setattr(torch.utils.data, 'DataLoader', lambda *a, **kw: DataLoader(
        *a, **dict(kw, pin_memory=False)))

    def run(name, resume=False):
        monkeypatch.setattr(sys, 'argv', ['train_full.py', '--data-path', str(tmp_path),
            '--manifest', str(manifest_path), '--output', str(tmp_path / name),
            '--epochs', '2', '--batch-size', '2', '--size', '32'] + (['--resume'] if resume else []))
        train_full.main()

    run('continuous')
    original_save = runtime.atomic_save

    def interrupt_after_epoch(state, path):
        original_save(state, path)
        if path.name == 'latest.pth' and state['epoch'] == 1:
            raise InterruptedError('Simulated process interruption')

    monkeypatch.setattr(runtime, 'atomic_save', interrupt_after_epoch)
    with pytest.raises(InterruptedError):
        run('resumed')
    monkeypatch.setattr(runtime, 'atomic_save', original_save)
    run('resumed', resume=True)
    continuous = torch.load(tmp_path / 'continuous/latest.pth', weights_only=False)
    resumed = torch.load(tmp_path / 'resumed/latest.pth', weights_only=False)
    for key in continuous['model_state_dict']:
        assert torch.equal(continuous['model_state_dict'][key], resumed['model_state_dict'][key])
    assert continuous['best_dice'] == resumed['best_dice']
    assert len(resumed['history']) == 2


def test_legacy_eval_matches_original_normalize_resize(tmp_path):
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from fullsup.runtime import SegmentationDataset
    image = np.random.RandomState(8).randint(0, 256, (41, 37, 3), dtype=np.uint8)
    mask = (image[:, :, 0] > 128).astype(np.uint8) * 255
    Image.fromarray(image).save(tmp_path / 'image.png')
    Image.fromarray(mask).save(tmp_path / 'mask.png')
    x, y, _ = SegmentationDataset(tmp_path, [['image.png', 'mask.png']], 32,
                                  preprocessing='legacy')[0]
    normalized = (image - 149.034) / 32.022
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min()) * 255
    old_x = TF.resize(torch.tensor(normalized).permute(2, 0, 1), [32, 32], antialias=False).float()
    old_y = TF.resize(torch.tensor(mask / 255).unsqueeze(0), [32, 32], antialias=False).float()
    assert torch.allclose(x, old_x, atol=1e-5)
    assert torch.equal(y, old_y)


def test_analysis_exports_csv_summary_and_worst_panels(tmp_path, monkeypatch):
    import sys
    import torch
    from PIL import Image
    import analyze_full
    import fullsup.runtime as runtime
    for folder in ('images', 'masks'):
        (tmp_path / 'val' / folder).mkdir(parents=True)
    for index in range(3):
        image = np.random.RandomState(index).randint(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(image).save(tmp_path / 'val/images' / '{}.png'.format(index))
        Image.fromarray((image[:, :, 0] > 128).astype(np.uint8) * 255).save(
            tmp_path / 'val/masks' / '{}.png'.format(index))
    def build():
        return torch.nn.Sequential(torch.nn.Conv2d(3, 1, 1), torch.nn.Sigmoid())
    ckpt = tmp_path / 'model.pth'
    torch.save(build().state_dict(), ckpt)
    monkeypatch.setattr(runtime, 'build_model', build)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, *a, **kw: self)
    out = tmp_path / 'analysis'
    monkeypatch.setattr(sys, 'argv', ['analyze_full.py', '--data-path', str(tmp_path),
        '--ckpt', str(ckpt), '--output', str(out), '--size', '32', '--worst-k', '2'])
    analyze_full.main()
    assert json.loads((out / 'summary.json').read_text())['n'] == 3
    assert len(list(out.glob('*.png'))) == 2
    assert len((out / 'per_image.csv').read_text().splitlines()) == 4
