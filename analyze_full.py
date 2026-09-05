"""Evaluate a trusted VM-UNet checkpoint and export per-image failure diagnostics."""
import argparse
import csv
import heapq
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', help='Use its val partition; otherwise use --split')
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--preprocessing', choices=['legacy', 'corrected'], default=None,
                        help='Default: checkpoint config; legacy for old state_dict checkpoints')
    parser.add_argument('--size', type=int, default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--worst-k', type=int, default=20)
    parser.add_argument('--boundary-tolerance', type=int, default=2, help='Pixels on resized evaluation grid')
    args = parser.parse_args()
    if args.manifest and args.split != 'val':
        parser.error('--manifest selects validation; do not combine with --split test')
    if args.worst_k < 0 or args.boundary_tolerance < 0 or (args.size is not None and (args.size <= 0 or args.size % 32)):
        parser.error('Invalid worst-k, tolerance or size (size must be a positive multiple of 32)')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from torch.utils.data import DataLoader
    from fullsup.data import load_manifest, pairs_in
    from fullsup.metrics import binary_metrics, summarize
    from fullsup.runtime import SegmentationDataset, build_model, load_weights, seed_everything, write_json
    seed_everything(42)
    pairs = load_manifest(args.data_path, args.manifest)['val'] if args.manifest else pairs_in(args.data_path, args.split)
    model = build_model()
    config = load_weights(model, args.ckpt)
    preprocessing = args.preprocessing or config.get('preprocessing', 'legacy')
    size = args.size or config.get('size', 256)
    if config and (size != config['size'] or preprocessing != config['preprocessing']):
        raise ValueError('Evaluation preprocessing/size must match the training checkpoint')
    model = model.cuda().eval()
    dataset = SegmentationDataset(args.data_path, pairs, size, False, preprocessing)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows, worst = [], []
    with torch.no_grad():
        for index, (image, target, names) in enumerate(loader):
            probability = model(image.cuda())[0, 0].cpu().numpy()
            if not np.isfinite(probability).all():
                raise FloatingPointError('Non-finite prediction for ' + names[0])
            pred, truth = probability >= 0.5, target[0, 0].numpy() >= 0.5
            row = dict(image=names[0], **binary_metrics(pred, truth, args.boundary_tolerance))
            rows.append(row)
            if args.worst_k:
                entry = (-row['dice'], index, pred, truth)
                if len(worst) < args.worst_k:
                    heapq.heappush(worst, entry)
                elif entry[0] > worst[0][0]:
                    heapq.heapreplace(worst, entry)
            if (index + 1) % 100 == 0:
                print('Evaluated {}/{}'.format(index + 1, len(dataset)), flush=True)
    with (output / 'per_image.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['dice']))
    summary = summarize(rows)
    summary.update(checkpoint=str(Path(args.ckpt).resolve()), preprocessing=preprocessing, size=size,
                   threshold=0.5, boundary_tolerance_px=args.boundary_tolerance,
                   evaluation_partition='manifest val' if args.manifest else args.split,
                   note='Pooled Dice matches legacy aggregation; macro Dice averages images. '
                        'HD95 is resized-grid pixels; one-empty cases undefined and counted separately. '
                        'Groups use GT area fraction: small <5%, medium <20%, large >=20%. '
                        'Diagnostic/validation results are not independent test evidence.')
    write_json(output / 'summary.json', summary)
    write_json(output / 'evaluation_pairs.json', pairs)
    for rank, (_, index, pred, truth) in enumerate(sorted(worst, key=lambda x: -x[0]), 1):
        with Image.open(Path(args.data_path) / pairs[index][0]) as im:
            raw = np.array(im.convert('RGB').resize((size, size)))
        gt = np.repeat((truth * 255).astype(np.uint8)[..., None], 3, axis=2)
        prediction = np.repeat((pred * 255).astype(np.uint8)[..., None], 3, axis=2)
        errors = (raw.astype(float) * 0.5).astype(np.uint8)
        errors[pred & ~truth] = [255, 0, 0]
        errors[~pred & truth] = [0, 100, 255]
        canvas = Image.new('RGB', (size * 4, size + 48), 'white')
        for k, panel in enumerate((raw, gt, prediction, errors)):
            canvas.paste(Image.fromarray(panel), (k * size, 48))
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 3), '{} | Dice {:.4f}'.format(pairs[index][0], rows[index]['dice']), fill='black')
        for k, title in enumerate(('Image', 'Ground truth', 'Prediction', 'Errors: red=FP blue=FN')):
            draw.text((k * size + 5, 25), title, fill='black')
        canvas.save(output / '{:02d}_{}.png'.format(rank, Path(pairs[index][0]).stem))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
