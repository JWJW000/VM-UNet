import copy
import json
import random

import numpy as np
import pytest
import torch
from PIL import Image

from fullsup.bcp import bcp_backward, load_unlabeled, region_loss, region_mask, sample_unlabeled
from fullsup.runtime import SegmentationDataset, segmentation_loss


def test_region_loss_and_sequential_gradients():
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 1, 1), torch.nn.Sigmoid())
    reference = copy.deepcopy(model)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    image, unlabeled = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    p = model(image)
    assert torch.allclose(region_loss(p, target, torch.ones_like(p)), segmentation_loss(p, target))
    assert region_loss(p, target, torch.zeros_like(p)) == 0
    actual = bcp_backward(model, teacher, image, target, unlabeled, seed=100)
    mask = region_mask(image, seed=100)
    assert mask.min() == 0 and mask.max() == 1
    pseudo = (teacher(unlabeled) >= 0.5).float()
    expected = 0
    for m in (mask, 1 - mask):
        p = reference(image * m + unlabeled * (1 - m))
        expected = expected + (region_loss(p, target, m) + .5 * region_loss(p, pseudo, 1 - m)) / 3
    expected.backward()
    assert actual == pytest.approx(expected.item())
    for a, b in zip(model.parameters(), reference.parameters()):
        assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
        assert torch.allclose(a.grad, b.grad)
    assert all(p.grad is None for p in teacher.parameters())


def test_unlabeled_partition_and_no_mask_access(tmp_path):
    for directory in ('train/images', 'train/masks', 'val/images', 'val/masks'):
        (tmp_path / directory).mkdir(parents=True)
    rng = np.random.RandomState(42)
    for name in ('train/images/a.png', 'train/images/u.png', 'val/images/v.png'):
        Image.fromarray(rng.randint(0, 256, (12, 12, 3), dtype=np.uint8)).save(tmp_path / name)
    manifest = dict(train=[['train/images/a.png', 'train/masks/a.png']],
                    val=[['val/images/v.png', 'val/masks/v.png']])
    path = tmp_path / 'split.json'
    split = dict(seed=42, labeled=[['a.png', 'a.png']], unlabeled=[['u.png', 'u.png']])
    path.write_text(json.dumps(split))
    pairs = load_unlabeled(tmp_path, manifest, path, 42)
    assert pairs == [['train/images/u.png', None]]
    ds = SegmentationDataset(tmp_path, pairs, 32, train=True)
    state = random.getstate()
    first = sample_unlabeled(ds, 2, 99)
    assert random.getstate() == state
    assert torch.equal(first, sample_unlabeled(ds, 2, 99))
    assert not (tmp_path / 'train/masks/u.png').exists()
    assert torch.isfinite(first).all() and first.shape == (2, 3, 32, 32)
    split['unlabeled'] = [['a.png', 'a.png']]
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match='overlap'):
        load_unlabeled(tmp_path, manifest, path, 42)
    split['unlabeled'] = [['../v.png', 'u.png']]
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match='basenames'):
        load_unlabeled(tmp_path, manifest, path, 42)
    split['unlabeled'] = [['u.png', 'u.png']]
    path.write_text(json.dumps(split))
    (tmp_path / 'train/images/u.png').write_bytes((tmp_path / 'val/images/v.png').read_bytes())
    with pytest.raises(ValueError, match='duplicates'):
        load_unlabeled(tmp_path, manifest, path, 42)


def test_bcp_options_reject_mixed_protocols(monkeypatch):
    import sys
    from train_full import parse_args
    base = ['train_full.py', '--data-path', 'data', '--manifest', 'split', '--output', 'out', '--bcp']
    for extra in ([], ['--init-checkpoint', 'init'], ['--init-checkpoint', 'init', '--unlabeled-split', 'ssl', '--teacher-weight', '1']):
        monkeypatch.setattr(sys, 'argv', base + extra)
        with pytest.raises(SystemExit):
            parse_args()
    monkeypatch.setattr(sys, 'argv', base + ['--init-checkpoint', 'init', '--unlabeled-split', 'ssl'])
    assert parse_args().bcp
