"""CPU-only extraction of old training logs; highlights Dice drops without guessing causes."""
import argparse
import csv
import json
import re
from pathlib import Path


def parse_log(text):
    rows = []
    for line in text.splitlines():
        match = re.search(r'val epoch:\s*(\d+),\s*loss:\s*([\d.eE+-]+)', line)
        if not match:
            continue
        dice = re.search(r'f1_or_dsc:\s*([\d.eE+-]+)', line)
        rows.append(dict(epoch=int(match[1]), val_loss=float(match[2]),
                         pooled_dice=float(dice[1]) if dice else None))
    measured = [r for r in rows if r['pooled_dice'] is not None]
    drops = [dict(previous_epoch=a['epoch'], epoch=b['epoch'],
                  drop_percentage_points=100 * (a['pooled_dice'] - b['pooled_dice']))
             for a, b in zip(measured, measured[1:])
             if b['epoch'] > a['epoch'] and a['pooled_dice'] - b['pooled_dice'] >= 0.05]
    return rows, drops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows, drops = parse_log(Path(args.log).read_text(errors='replace'))
    if not rows:
        raise ValueError('No legacy validation records found')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'validation.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'val_loss', 'pooled_dice'])
        writer.writeheader()
        writer.writerows(rows)
    report = dict(log=args.log, validation_records=len(rows), drops_at_least_5_points=drops,
                  note='Drops flag a diagnostic task, not a causal conclusion. '
                       'Check LR, resume, data, non-finite gradients and checkpoint provenance.')
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
