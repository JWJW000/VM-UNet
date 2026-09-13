#!/usr/bin/env bash
# One fixed 10%-label experiment: smoke -> supervised continuation -> BCP adaptation.
set -euo pipefail
cd "$(dirname "$0")"
python - <<'CHECK'
import hashlib
import json
from pathlib import Path
import torch
checks = {
    'results/full_fewlabel_r0p1_s42/best.pth': '05e429b4f8c5dfed729fe3068fbb0c9a5a827de8fd1c76956acca7f088e8d1fc',
    'results/full_fewlabel_r0p1_s42/manifest.json': 'd1d4f587a50546f114caa158ca8261f89b43f93f16623598a7fa69df7707e0dd',
    'splits/isic18_r0p1_s42.json': '621d499b3cd6527c046ff99aaf84e11f761797f9f7aae4236bc98523d2d74c00',
}
for name, digest in checks.items():
    p = Path(name)
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
        raise ValueError('Missing/mismatched frozen input; restore original artifact: ' + name)
manifest = json.loads(Path('results/full_fewlabel_r0p1_s42/manifest.json').read_text())
assert len(manifest['train']) == 189 and len(manifest['val']) == 808
assert torch.cuda.is_available(), 'Activate the original CUDA/Mamba environment'
for name in ('bcp10_smoke_s42', 'bcp10_control_s42', 'bcp10_method_s42',
             'diagnosis_bcp10_control_s42', 'diagnosis_bcp10_method_s42'):
    if Path('results', name).exists():
        raise FileExistsError('Existing output: ' + name + '. Inspect/resume it; do not rerun this chain.')
print('Frozen inputs verified. Each formal arm: 300 x ceil(189/32) = 1800 updates.', flush=True)
CHECK
args=(--model vmunet --data-path data/isic2018
      --manifest results/full_fewlabel_r0p1_s42/manifest.json
      --init-checkpoint results/full_fewlabel_r0p1_s42/best.pth
      --unlabeled-split splits/isic18_r0p1_s42.json
      --expected-initial-dice 0.8698412827556765
      --seed 42 --size 256 --batch-size 32 --lr 0.0001 --weight-decay 0.01
      --eta-min 0.00001 --t-max 300 --preprocessing corrected --gpu 0)
# An isolated six-update smoke run. Neither arm starts from its weights.
python -u train_full.py "${args[@]}" --bcp --epochs 1 --output results/bcp10_smoke_s42
python - <<'CHECK'
import torch
from fullsup.runtime import build_model
state = torch.load('results/bcp10_smoke_s42/latest.pth', map_location='cpu', weights_only=False)
model = build_model().cuda().eval()
with torch.no_grad():
    for key in ('model_state_dict', 'teacher_state_dict'):
        model.load_state_dict(state[key], strict=True)
        p = model(torch.zeros(1, 3, 256, 256, device='cuda'))
        assert p.shape == (1, 1, 256, 256) and torch.isfinite(p).all()
assert state['epoch'] == 1 and state['history'][-1]['optimizer_updates'] == 6
print('Student and EMA checkpoint reload passed. Starting paired runs.', flush=True)
CHECK
python -u train_full.py "${args[@]}" --epochs 300 --output results/bcp10_control_s42
python -u analyze_full.py --ckpt results/bcp10_control_s42/best.pth \
    --data-path data/isic2018 --manifest results/full_fewlabel_r0p1_s42/manifest.json \
    --output results/diagnosis_bcp10_control_s42 --gpu 0
python -u train_full.py "${args[@]}" --bcp --epochs 300 --output results/bcp10_method_s42
python -u analyze_full.py --ckpt results/bcp10_method_s42/best.pth \
    --data-path data/isic2018 --manifest results/full_fewlabel_r0p1_s42/manifest.json \
    --output results/diagnosis_bcp10_method_s42 --gpu 0
python - <<'REPORT'
import csv
import json
from pathlib import Path
import torch
results = {}
for arm in ('control', 'method'):
    directory = Path('results/bcp10_' + arm + '_s42')
    rows = list(csv.DictReader((directory / 'metrics.csv').open()))
    assert [int(row['epoch']) for row in rows] == list(range(1, 301))
    assert int(rows[-1]['optimizer_updates']) == 1800
    best = torch.load(directory / 'best.pth', map_location='cpu', weights_only=False)
    results[arm] = dict(best_epoch=best['epoch'], **best['metrics'],
                       last_dice=float(rows[-1]['pooled_dice']),
                       epoch_seconds=sum(float(row['seconds']) for row in rows))
d = results['method']['pooled_dice']
results['delta_dice_vs_initial'] = d - 0.8698412827556765
results['delta_dice_vs_control'] = d - results['control']['pooled_dice']
results['screen_passed'] = (d >= 0.8798412827556765 and d > results['control']['pooled_dice']
                          and results['method']['pooled_iou'] > results['control']['pooled_iou'])
results['note'] = 'Single-seed development screen; no automatic new training. Review macro/boundary and cases.'
Path('results/bcp10_comparison.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
REPORT
