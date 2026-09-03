"""Scan-direction utilities for SS2D.

VM-UNet's SS2D always sums four Cross-Scan streams (row, column, and their
flips). These helpers keep that 4-way sum when every direction is on, and
re-scale when some directions are dropped so feature magnitude stays comparable.
"""
import torch


N_SCAN_DIRS = 4


def scan_mask_weights(scan_mask, batch_size, device, dtype):
    """Turn a 4-dir on/off mask into per-direction scale factors.

    Args:
        scan_mask: None, length-4 sequence, or tensor shaped (4,) / (B, 4).
        batch_size: B of the scan tensors.
        device, dtype: match the scan activations.

    Returns:
        None if scan_mask is None (caller should use the original 4-way sum).
        Otherwise a tensor of shape (B, 4) whose rows sum to N_SCAN_DIRS.
    """
    if scan_mask is None:
        return None
    mask = torch.as_tensor(scan_mask, device=device, dtype=dtype)
    if mask.dim() == 1:
        if mask.numel() != N_SCAN_DIRS:
            raise ValueError(f'scan_mask length must be {N_SCAN_DIRS}, got {tuple(mask.shape)}')
        mask = mask.view(1, N_SCAN_DIRS).expand(batch_size, -1).clone()
    elif mask.dim() == 2:
        if mask.shape[-1] != N_SCAN_DIRS:
            raise ValueError(f'scan_mask last dim must be {N_SCAN_DIRS}, got {tuple(mask.shape)}')
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1).clone()
        elif mask.shape[0] != batch_size:
            raise ValueError(f'scan_mask batch {mask.shape[0]} != activation batch {batch_size}')
    else:
        raise ValueError(f'scan_mask must be 1D or 2D, got {tuple(mask.shape)}')

    zero_rows = mask.sum(dim=-1) <= 0
    if torch.any(zero_rows):
        mask = mask.clone()
        mask[zero_rows] = 1

    n_active = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    return mask * (N_SCAN_DIRS / n_active)


def combine_scan_directions(direction_outputs, scan_mask=None):
    """Sum four SS2D streams, optionally dropping directions.

    Args:
        direction_outputs: length-4 sequence of tensors with the same shape.
        scan_mask: None for the original y1+y2+y3+y4; otherwise on/off weights.

    Returns:
        Combined tensor, same shape as each direction output.
    """
    if len(direction_outputs) != N_SCAN_DIRS:
        raise ValueError(f'expected {N_SCAN_DIRS} direction tensors, got {len(direction_outputs)}')
    y0 = direction_outputs[0]
    weights = scan_mask_weights(scan_mask, y0.shape[0], y0.device, y0.dtype)
    if weights is None:
        y = direction_outputs[0]
        for extra in direction_outputs[1:]:
            y = y + extra
        return y

    y = torch.zeros_like(y0)
    leading = (weights.shape[0],) + (1,) * (y0.dim() - 1)
    for i, yi in enumerate(direction_outputs):
        y = y + yi * weights[:, i].view(leading)
    return y


def sample_scan_mask(n_keep=2, device='cpu', dtype=torch.float32):
    """Sample one on/off mask that keeps ``n_keep`` of the four directions."""
    if not 1 <= n_keep <= N_SCAN_DIRS:
        raise ValueError(f'n_keep must be in [1, {N_SCAN_DIRS}], got {n_keep}')
    idx = torch.randperm(N_SCAN_DIRS, device=device)
    mask = torch.zeros(N_SCAN_DIRS, device=device, dtype=dtype)
    mask[idx[:n_keep]] = 1
    return mask


def complementary_scan_mask(mask):
    """Flip a 4-dir on/off mask. All-zero / all-one masks are left unchanged."""
    mask = torch.as_tensor(mask)
    flipped = (mask <= 0).to(mask.dtype)
    if torch.all(flipped == 0) or torch.all(flipped == 1):
        return mask.clone() if torch.is_tensor(mask) else mask
    return flipped


def sample_complementary_pair(device='cpu', dtype=torch.float32):
    """Two complementary 2-dir masks covering all four Cross-Scan directions."""
    mask_a = sample_scan_mask(n_keep=2, device=device, dtype=dtype)
    mask_b = complementary_scan_mask(mask_a)
    return mask_a, mask_b
