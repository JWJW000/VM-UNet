"""Reproducible labeled/unlabeled splits for scan-aware SSL."""
import json
import os
import random


def list_train_pairs(data_path):
    """Return sorted (image_name, mask_name) pairs from ``train/``."""
    img_dir = os.path.join(data_path, 'train', 'images')
    mask_dir = os.path.join(data_path, 'train', 'masks')
    images = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))
    if len(images) != len(masks):
        raise RuntimeError(
            f'train images ({len(images)}) and masks ({len(masks)}) counts differ under {data_path}'
        )
    return list(zip(images, masks))


def split_path(split_dir, dataset, labeled_ratio, seed):
    ratio_tag = str(labeled_ratio).replace('.', 'p')
    return os.path.join(split_dir, f'{dataset}_r{ratio_tag}_s{seed}.json')


def make_labeled_split(data_path, dataset, labeled_ratio, seed, split_dir='./splits'):
    """Create or load a frozen split. Never reshuffles an existing file."""
    if not 0 < labeled_ratio <= 1:
        raise ValueError(f'labeled_ratio must be in (0, 1], got {labeled_ratio}')
    os.makedirs(split_dir, exist_ok=True)
    path = split_path(split_dir, dataset, labeled_ratio, seed)
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f), path

    pairs = list_train_pairs(data_path)
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    n_labeled = max(1, int(round(len(pairs) * labeled_ratio)))
    n_labeled = min(n_labeled, len(pairs))
    labeled_idx = sorted(order[:n_labeled])
    unlabeled_idx = sorted(order[n_labeled:])

    payload = {
        'dataset': dataset,
        'data_path': os.path.abspath(data_path),
        'labeled_ratio': labeled_ratio,
        'seed': seed,
        'n_total': len(pairs),
        'n_labeled': len(labeled_idx),
        'n_unlabeled': len(unlabeled_idx),
        'labeled': [list(pairs[i]) for i in labeled_idx],
        'unlabeled': [list(pairs[i]) for i in unlabeled_idx],
    }
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')
    os.replace(tmp, path)
    return payload, path
