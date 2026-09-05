"""Explicit image/mask pairing and deterministic manifest generation."""
import hashlib
import json
import random
from pathlib import Path


def pairs_in(root, split):
    root = Path(root)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    images = sorted(p for p in (root / split / 'images').iterdir()
                    if p.suffix.lower() in extensions)
    masks = {}
    for p in (root / split / 'masks').iterdir():
        if p.suffix.lower() not in extensions:
            continue
        key = p.stem[:-13] if p.stem.endswith('_segmentation') else p.stem
        if key in masks:
            raise ValueError('Ambiguous mask ID: ' + key)
        masks[key] = p
    if not images or len({p.stem for p in images}) != len(images):
        raise ValueError('Empty image directory or duplicate image IDs: ' + split)
    if set(masks) != {p.stem for p in images}:
        raise ValueError('Images and masks must match by stem (optional _segmentation suffix)')
    return [[str(p.relative_to(root)), str(masks[p.stem].relative_to(root))] for p in images]


def validate_manifest(root, manifest):
    root = Path(root).resolve()
    seen_paths, seen_ids, seen_content = set(), set(), {}
    for split in ('train', 'val'):
        if not manifest.get(split):
            raise ValueError('Empty manifest partition: ' + split)
        for pair in manifest[split]:
            if len(pair) != 2:
                raise ValueError('Expected image/mask pairs')
            for name in pair:
                path = (root / name).resolve()
                if root not in path.parents or not path.is_file():
                    raise ValueError('Invalid dataset path: ' + name)
            image_path = (root / pair[0]).resolve()
            image_id = image_path.stem
            if image_path in seen_paths or image_id in seen_ids:
                raise ValueError('Repeated image/path across manifest: ' + pair[0])
            seen_paths.add(image_path)
            seen_ids.add(image_id)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if digest in seen_content and seen_content[digest] != split:
                raise ValueError('Identical image bytes across train/val: ' + pair[0])
            seen_content[digest] = split
    # This detects exact copies only; patient/lesion grouping requires metadata.


def make_manifest(root, val_fraction=0.0, seed=42):
    train = pairs_in(root, 'train')
    if val_fraction:
        if not 0 < val_fraction < 1:
            raise ValueError('val_fraction must be in (0, 1)')
        random.Random(seed).shuffle(train)
        n = max(1, round(len(train) * val_fraction))
        train, val = sorted(train[n:]), sorted(train[:n])
        protocol = 'internal_train_holdout; original val excluded'
    else:
        val = pairs_in(root, 'val')
        protocol = 'legacy_train_val_development; not independent test'
    result = dict(version=1, seed=seed, protocol=protocol, train=train, val=val)
    validate_manifest(root, result)
    return result


def load_manifest(root, path):
    manifest = json.loads(Path(path).read_text())
    validate_manifest(root, manifest)
    return manifest
