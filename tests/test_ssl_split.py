import json
from pathlib import Path

from scan_ssl.splits import list_train_pairs, make_labeled_split


def _fake_isic(root: Path, n=10):
    img_dir = root / 'train' / 'images'
    mask_dir = root / 'train' / 'masks'
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    names = []
    for i in range(n):
        name = f'img_{i:03d}.png'
        (img_dir / name).write_bytes(b'')
        (mask_dir / name).write_bytes(b'')
        names.append(name)
    return names


def test_list_train_pairs_sorted(tmp_path):
    _fake_isic(tmp_path, n=5)
    pairs = list_train_pairs(str(tmp_path) + '/')
    assert len(pairs) == 5
    assert pairs[0][0] == 'img_000.png'


def test_make_labeled_split_is_deterministic(tmp_path):
    _fake_isic(tmp_path, n=10)
    split_dir = tmp_path / 'splits'
    a, path_a = make_labeled_split(str(tmp_path) + '/', 'isic18', 0.3, seed=0, split_dir=str(split_dir))
    b, path_b = make_labeled_split(str(tmp_path) + '/', 'isic18', 0.3, seed=0, split_dir=str(split_dir))
    assert path_a == path_b
    assert a['labeled'] == b['labeled']
    assert a['n_labeled'] == 3
    assert a['n_unlabeled'] == 7
    labeled_set = {row[0] for row in a['labeled']}
    unlabeled_set = {row[0] for row in a['unlabeled']}
    assert labeled_set.isdisjoint(unlabeled_set)
    assert labeled_set | unlabeled_set == {f'img_{i:03d}.png' for i in range(10)}


def test_existing_split_is_not_reshuffled(tmp_path):
    _fake_isic(tmp_path, n=8)
    split_dir = tmp_path / 'splits'
    first, path = make_labeled_split(str(tmp_path) + '/', 'isic18', 0.5, seed=1, split_dir=str(split_dir))
    payload = json.loads(Path(path).read_text())
    payload['labeled'] = payload['labeled'][:1]
    payload['n_labeled'] = 1
    Path(path).write_text(json.dumps(payload))
    second, _ = make_labeled_split(str(tmp_path) + '/', 'isic18', 0.5, seed=1, split_dir=str(split_dir))
    assert second['n_labeled'] == 1


def test_sigmoid_rampup_bounds():
    from scan_ssl.losses import sigmoid_rampup
    assert sigmoid_rampup(0, 10) < 0.1
    assert sigmoid_rampup(10, 10) == 1.0
    assert sigmoid_rampup(3, 0) == 1.0


def test_confidence_mse_ignores_uncertain_pixels():
    import torch
    from scan_ssl.losses import confidence_mse
    student = torch.tensor([[[[0.9, 0.5]]]])
    teacher = torch.tensor([[[[1.0, 0.5]]]])
    loss = confidence_mse(student, teacher, conf_lo=0.2, conf_hi=0.8)
    assert torch.allclose(loss, torch.tensor(0.01))
