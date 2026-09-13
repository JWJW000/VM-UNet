"""Binary VM-UNet adaptation of BCP (CVPR 2023), not a new architecture.

Reference: https://github.com/DeepMed-Lab-ECNU/BCP
Reuse the supervised checkpoint; no ACDC-specific class/connected-component rules.
"""
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F


def load_unlabeled(root, manifest, split_path, seed):
    """Check the frozen SSL partition; never open an unlabeled ground-truth mask."""
    root = Path(root).resolve()
    split = json.loads(Path(split_path).read_text())
    if split.get('seed') != seed:
        raise ValueError('SSL split seed differs')
    partitions = {}
    for name in ('labeled', 'unlabeled'):
        pairs = split.get(name, [])
        if not pairs or any(not isinstance(p, list) or len(p) != 2 for p in pairs):
            raise ValueError('SSL split requires nonempty image/mask filename pairs')
        if any(not isinstance(n, str) or Path(n).name != n or n in ('.', '..') for p in pairs for n in p):
            raise ValueError('SSL filenames must be basenames')
        if len({p[0] for p in pairs}) != len(pairs):
            raise ValueError('Repeated image in SSL partition')
        partitions[name] = pairs
    labeled = {('train/images/' + a, 'train/masks/' + b) for a, b in partitions['labeled']}
    if labeled != set(map(tuple, manifest['train'])):
        raise ValueError('SSL labeled pairs differ from supervised manifest')
    names_l = {a for a, _ in partitions['labeled']}
    names_u = {a for a, _ in partitions['unlabeled']}
    if names_l & names_u:
        raise ValueError('Labeled/unlabeled overlap')
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    all_names = {p.name for p in (root / 'train/images').iterdir() if p.suffix.lower() in extensions}
    if names_l | names_u != all_names:
        raise ValueError('SSL split must cover the training image directory exactly')
    known_paths = {(root / a).resolve() for a, _ in manifest['train'] + manifest['val']}
    known_bytes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in known_paths}
    pairs = []
    for name, _ in partitions['unlabeled']:
        relative = 'train/images/' + name
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or path in known_paths:
            raise ValueError('Invalid/overlapping unlabeled image: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() in known_bytes:
            raise ValueError('Unlabeled image duplicates labeled/validation bytes: ' + name)
        pairs.append([relative, None])
    return pairs


def sample_unlabeled(dataset, count, seed):
    """Step-seeded sampling keeps labeled augmentation RNG unchanged and resumable."""
    state = random.getstate()
    try:
        random.seed(seed)
        indices = (random.sample(range(len(dataset)), count) if count <= len(dataset)
                   else random.choices(range(len(dataset)), k=count))
        return torch.stack([dataset[i][0] for i in indices])
    finally:
        random.setstate(state)


def region_mask(image, seed):
    """One random 2/3-height by 2/3-width rectangle, shared across the batch."""
    h, w = image.shape[-2:]
    ph, pw = max(1, h * 2 // 3), max(1, w * 2 // 3)
    generator = torch.Generator().manual_seed(seed)
    y = int(torch.randint(h - ph + 1, (), generator=generator))
    x = int(torch.randint(w - pw + 1, (), generator=generator))
    mask = image.new_ones((len(image), 1, h, w))
    mask[:, :, y:y + ph, x:x + pw] = 0
    return mask


def region_loss(probability, target, mask):
    """BCE + soft Dice restricted to a region; an empty region contributes zero."""
    count = mask.flatten(1).sum(1)
    active = count > 0
    if not active.any():
        return probability.sum() * 0
    bce = (F.binary_cross_entropy(probability, target, reduction='none') * mask).flatten(1).sum(1)
    p, t = probability * mask, target * mask
    dice = 1 - (2 * (p * t).flatten(1).sum(1) + 1) / (p.flatten(1).sum(1) + t.flatten(1).sum(1) + 1)
    return (bce / count.clamp_min(1) + dice)[active].mean()


def bcp_backward(student, teacher, labeled, target, unlabeled, seed, pseudo_weight=0.5):
    """Accumulate both mixed-view gradients without retaining two activation graphs.

    No optimizer step here. Teacher has no gradients and is updated by EMA afterwards.
    """
    with torch.no_grad():
        probability = teacher(unlabeled)
        if not torch.isfinite(probability).all():
            raise FloatingPointError('Non-finite BCP teacher prediction')
        pseudo = (probability >= 0.5).to(target.dtype)
    mask = region_mask(labeled, seed)
    total = 0.0
    for labeled_mask in (mask, 1 - mask):
        mixed = labeled * labeled_mask + unlabeled * (1 - labeled_mask)
        probability = student(mixed)
        if not torch.isfinite(probability).all():
            raise FloatingPointError('Non-finite BCP student prediction')
        loss = (region_loss(probability, target, labeled_mask)
                + pseudo_weight * region_loss(probability, pseudo, 1 - labeled_mask)) / (1 + pseudo_weight)
        loss = loss / 2  # Mean of the two directions, one optimizer update.
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite BCP loss')
        loss.backward()
        total += loss.item()
    return total
