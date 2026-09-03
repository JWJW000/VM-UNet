"""SSL losses for scan-aware consistency."""
import numpy as np
import torch
import torch.nn.functional as F


def sigmoid_rampup(current, rampup_length):
    """Gaussian ramp-up used by Mean Teacher / UA-MT."""
    if rampup_length <= 0:
        return 1.0
    current = np.clip(float(current), 0.0, float(rampup_length))
    phase = 1.0 - current / float(rampup_length)
    return float(np.exp(-5.0 * phase * phase))


def confidence_mse(student_prob, teacher_prob, conf_lo=0.2, conf_hi=0.8):
    """Pixel MSE on teacher-confident locations.

    Teacher/student outputs are already in [0, 1] for binary VM-UNet.
    Pixels with teacher probability in (conf_lo, conf_hi) are ignored.
    """
    teacher = teacher_prob.detach()
    confident = (teacher >= conf_hi) | (teacher <= conf_lo)
    if confident.dtype != torch.bool:
        confident = confident.bool()
    if confident.sum() == 0:
        return student_prob.mean() * 0.0
    diff = (student_prob - teacher) ** 2
    return diff[confident].mean()


def prediction_mse(pred_a, pred_b):
    """Symmetric MSE between two student views (direction consistency)."""
    return F.mse_loss(pred_a, pred_b.detach()) + F.mse_loss(pred_b, pred_a.detach())
