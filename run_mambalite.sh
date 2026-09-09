#!/usr/bin/env bash
# Run from the repository root in the existing CUDA/Mamba environment.
set -euo pipefail
cd "$(dirname "$0")"
python - <<'CHECK'
import hashlib
from pathlib import Path
import torch
from mamba_ssm import Mamba
assert torch.cuda.is_available(), 'Activate the existing CUDA/Mamba environment first'
p = Path('splits/full_isic18_legacy.json')
assert p.is_file(), 'Missing frozen baseline manifest; restore it from the B0 run'
assert hashlib.sha256(p.read_bytes()).hexdigest() == '5e5790bc5a5f8410c908b4ae19683f03b041d94110199a8c4ed2ecb860707238', 'Manifest differs from B0'
from importlib.metadata import version
print({name: version(name) for name in ('torch', 'torchvision', 'mamba-ssm', 'causal-conv1d', 'timm')}, flush=True)
print('CUDA:', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0), flush=True)
CHECK
# Separate fresh outputs prevent overwriting or accidentally resuming old work.
args=(--model mambalite --data-path data/isic2018
      --manifest splits/full_isic18_legacy.json --seed 43 --size 256
      --batch-size 32 --lr 0.001 --weight-decay 0.01 --eta-min 0.00001
      --t-max 300 --preprocessing corrected --gpu 0)
python -u train_full.py "${args[@]}" --epochs 1 --output results/full_mambalite_smoke_s43
# Verify the real smoke checkpoint can be restored before committing 300 epochs.
python - <<'CHECK'
import torch
from fullsup.runtime import build_model, load_weights
model = build_model(model_name='mambalite').cuda().eval()
load_weights(model, 'results/full_mambalite_smoke_s43/best.pth')
with torch.no_grad():
    output = model(torch.zeros(1, 3, 256, 256, device='cuda'))
assert output.shape == (1, 1, 256, 256) and torch.isfinite(output).all()
print('CUDA checkpoint reload passed; starting fresh seed43 training.', flush=True)
CHECK
python -u train_full.py "${args[@]}" --epochs 300 --output results/full_mambalite_s43
python -u analyze_full.py --ckpt results/full_mambalite_s43/best.pth \
    --data-path data/isic2018 --manifest splits/full_isic18_legacy.json \
    --output results/diagnosis_full_mambalite_s43 --gpu 0
