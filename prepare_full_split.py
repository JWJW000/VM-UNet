"""Freeze full-label training/validation pairs without requiring CUDA or torch."""
import argparse
import json
from pathlib import Path
from fullsup.data import make_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--val-fraction', type=float, default=0.0,
                        help='0 preserves repository train/val; >0 holds out from train')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    manifest = make_manifest(args.data_path, args.val_fraction, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(manifest, f, indent=2)
    print('{}: train={}, val={}, {}'.format(path, len(manifest['train']),
                                          len(manifest['val']), manifest['protocol']))


if __name__ == '__main__':
    main()
