"""EMA teacher update. Teacher stays in eval; only buffers/params are copied."""
import torch


@torch.no_grad()
def update_ema(student, teacher, decay):
    student_params = dict(student.named_parameters())
    for name, t_param in teacher.named_parameters():
        s_param = student_params[name]
        t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)

    student_buffers = dict(student.named_buffers())
    for name, t_buf in teacher.named_buffers():
        s_buf = student_buffers.get(name)
        if s_buf is None or s_buf.dtype != t_buf.dtype or s_buf.shape != t_buf.shape:
            continue
        t_buf.data.copy_(s_buf.data)
