import torch

from models.vmunet.scan_utils import (
    combine_scan_directions,
    complementary_scan_mask,
    sample_complementary_pair,
    sample_scan_mask,
    scan_mask_weights,
)


def test_combine_none_matches_plain_sum():
    dirs = [torch.ones(2, 3, 4) * (i + 1) for i in range(4)]
    out = combine_scan_directions(dirs, scan_mask=None)
    assert torch.allclose(out, dirs[0] + dirs[1] + dirs[2] + dirs[3])


def test_all_ones_mask_matches_plain_sum():
    dirs = [torch.randn(2, 3, 4) for _ in range(4)]
    out = combine_scan_directions(dirs, scan_mask=torch.ones(4))
    assert torch.allclose(out, dirs[0] + dirs[1] + dirs[2] + dirs[3], atol=1e-6)


def test_two_dir_mask_rescales_to_four_way_magnitude():
    dirs = [torch.ones(1, 2, 3) for _ in range(4)]
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    out = combine_scan_directions(dirs, scan_mask=mask)
    # two active ones, scaled by 4/2 = 2, so 1+1 then *2? weights = mask * (4/2) = [2,2,0,0]
    # y = 1*2 + 1*2 = 4, same as four-way sum of ones.
    assert torch.allclose(out, torch.full((1, 2, 3), 4.0))


def test_zero_mask_falls_back_to_all_directions():
    dirs = [torch.ones(1, 1, 1) * (i + 1) for i in range(4)]
    out = combine_scan_directions(dirs, scan_mask=torch.zeros(4))
    assert torch.allclose(out, torch.tensor([[[10.0]]]))


def test_batch_mask_applies_per_sample():
    dirs = [torch.ones(2, 1, 1) for _ in range(4)]
    mask = torch.tensor([[1.0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0]])
    out = combine_scan_directions(dirs, scan_mask=mask)
    assert torch.allclose(out[0], torch.tensor([[4.0]]))
    assert torch.allclose(out[1], torch.tensor([[4.0]]))


def test_sample_scan_mask_keeps_requested_count():
    torch.manual_seed(0)
    mask = sample_scan_mask(n_keep=2)
    assert mask.shape == (4,)
    assert int(mask.sum().item()) == 2


def test_complementary_pair_is_partition_of_four():
    torch.manual_seed(1)
    a, b = sample_complementary_pair()
    assert int((a + b).sum().item()) == 4
    assert torch.all((a * b) == 0)
    assert torch.equal(complementary_scan_mask(a), b)


def test_scan_mask_weights_rejects_bad_shape():
    try:
        scan_mask_weights(torch.ones(3), batch_size=1, device='cpu', dtype=torch.float32)
        assert False, 'expected ValueError'
    except ValueError:
        pass
