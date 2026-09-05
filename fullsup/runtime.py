"""Shared data, loss and inference code for controlled full-supervision runs."""
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SegmentationDataset(Dataset):
    def __init__(self, root, pairs, size=256, train=False, preprocessing='corrected'):
        self.root, self.pairs = Path(root), pairs
        self.size, self.train, self.preprocessing = size, train, preprocessing
        # Legacy rotation samples one angle on construction, corrected samples per image.
        self.legacy_angle = random.uniform(0, 360) if train else 0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_name, mask_name = self.pairs[index]
        with Image.open(self.root / image_name) as im:
            image = np.array(im.convert('RGB'), dtype=np.float64)
        with Image.open(self.root / mask_name) as im:
            mask = np.array(im.convert('L'), dtype=np.float64) / 255.0
        if image.shape[:2] != mask.shape:
            raise ValueError('Image/mask dimensions differ: ' + image_name)
        # Algebraically identical to original myNormalize (its affine stats cancel).
        span = float(image.max() - image.min())
        image = (image - image.min()) / span * 255.0 if span else np.zeros_like(image)
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        if self.train:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)
            if random.random() < 0.5:
                angle = self.legacy_angle if self.preprocessing == 'legacy' else random.uniform(0, 360)
                image, mask = TF.rotate(image, angle), TF.rotate(mask, angle)
        image = TF.resize(image, [self.size, self.size], interpolation=InterpolationMode.BILINEAR,
                          antialias=False)
        mask = TF.resize(mask, [self.size, self.size], interpolation=(
            InterpolationMode.BILINEAR if self.preprocessing == 'legacy' else InterpolationMode.NEAREST),
            antialias=False)
        return image.float(), mask.float(), image_name


def segmentation_loss(prob, target, boundary_weight=0.0):
    """Original BCE + per-image soft Dice, optionally emphasizing a 5x5 GT edge band.

    Boundary weighting is a diagnostic control, not a claimed new architecture.
    """
    bce = F.binary_cross_entropy(prob, target, reduction='none')
    if boundary_weight:
        hard = (target >= 0.5).float()
        band = F.max_pool2d(hard, 5, 1, 2) + F.max_pool2d(-hard, 5, 1, 2)
        weights = 1 + boundary_weight * band
        bce_loss = (bce * weights).sum() / weights.sum()
    else:
        bce_loss = bce.mean()
    p, t = prob.flatten(1), target.flatten(1)
    dice_loss = 1 - ((2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()
    return bce_loss + dice_loss


def build_model(pretrained=None):
    from models.vmunet.vmunet import VMUNet
    model = VMUNet(input_channels=3, num_classes=1, depths=[2, 2, 2, 2],
                   depths_decoder=[2, 2, 2, 1], drop_path_rate=0.2,
                   load_ckpt_path=pretrained)
    if pretrained:
        model.load_from()
    return model


def load_weights(model, checkpoint):
    # Only load trusted, user-owned checkpoints: legacy torch.load uses pickle.
    try:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location='cpu')
    payload = state.get('model_state_dict', state)
    # Legacy best.pth may include thop profiling buffers.
    cleaned = {k: v for k, v in payload.items()
               if not k.endswith('total_ops') and not k.endswith('total_params')}
    model.load_state_dict(cleaned, strict=False)
    return state.get('config', {}) if isinstance(state, dict) else {}


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


@torch.no_grad()
def validate(model, loader, device, threshold=0.5):
    model.eval()
    tp = fp = fn = 0
    dice_sum = loss_sum = count = 0
    for image, target, _ in loader:
        image, target = image.to(device), target.to(device)
        prob = model(image)
        if not torch.isfinite(prob).all():
            raise FloatingPointError('Non-finite validation prediction')
        loss_sum += segmentation_loss(prob, target).item() * len(image)
        pred, truth = prob >= threshold, target >= 0.5
        a = (pred & truth).flatten(1).sum(1)
        b = (pred & ~truth).flatten(1).sum(1)
        c = (~pred & truth).flatten(1).sum(1)
        denom = 2 * a + b + c
        dice_sum += torch.where(denom > 0, 2 * a / denom.clamp_min(1),
                                torch.ones_like(denom, dtype=torch.float)).sum().item()
        tp += a.sum().item()
        fp += b.sum().item()
        fn += c.sum().item()
        count += len(image)
    if not count:
        raise ValueError('Empty validation loader')
    from fullsup.metrics import overlap
    pooled = overlap(tp, fp, fn)
    return dict(val_loss=loss_sum / count, pooled_dice=pooled['dice'],
                pooled_iou=pooled['iou'], macro_dice=dice_sum / count)
