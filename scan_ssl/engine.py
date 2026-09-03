import copy
from itertools import cycle

import numpy as np
import torch

from models.vmunet.scan_utils import complementary_scan_mask, sample_scan_mask
from scan_ssl.ema import update_ema
from scan_ssl.losses import confidence_mse, prediction_mse, sigmoid_rampup


def build_teacher(student):
    teacher = copy.deepcopy(student)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    return teacher


def _to_cuda(batch):
    images, targets = batch
    images = images.cuda(non_blocking=True).float()
    targets = targets.cuda(non_blocking=True).float()
    return images, targets


def train_one_epoch_ssl(
    labeled_loader,
    unlabeled_loader,
    student,
    teacher,
    criterion,
    optimizer,
    scheduler,
    epoch,
    step,
    logger,
    config,
    writer,
):
    student.train()
    teacher.eval()

    n_iter = max(len(labeled_loader), len(unlabeled_loader))
    labeled_iter = cycle(labeled_loader)
    unlabeled_iter = cycle(unlabeled_loader)

    cons_w = config.cons_weight * sigmoid_rampup(epoch, config.rampup_epochs)
    dir_w = config.dir_weight * sigmoid_rampup(epoch, config.rampup_epochs)
    if not config.use_dir_consistency:
        dir_w = 0.0

    loss_meter = []
    sup_meter = []
    cons_meter = []
    dir_meter = []

    for it in range(n_iter):
        step += 1
        optimizer.zero_grad()

        xl, yl = _to_cuda(next(labeled_iter))
        xu, _yu = _to_cuda(next(unlabeled_iter))

        out_l = student(xl)
        loss_sup = criterion(out_l, yl)

        device = xu.device
        if config.use_scan_dropout:
            mask_a = sample_scan_mask(n_keep=config.scan_keep, device=device)
        else:
            mask_a = torch.ones(4, device=device)

        with torch.no_grad():
            teacher_u = teacher(xu)

        student_u = student(xu, scan_mask=mask_a)
        loss_cons = confidence_mse(
            student_u, teacher_u, conf_lo=config.conf_lo, conf_hi=config.conf_hi
        )

        loss_dir = student_u.mean() * 0.0
        if dir_w > 0:
            mask_b = complementary_scan_mask(mask_a)
            student_u_b = student(xu, scan_mask=mask_b)
            loss_dir = prediction_mse(student_u, student_u_b)

        loss = loss_sup + cons_w * loss_cons + dir_w * loss_dir
        loss.backward()
        optimizer.step()
        update_ema(student, teacher, config.ema_decay)

        loss_meter.append(loss.item())
        sup_meter.append(loss_sup.item())
        cons_meter.append(loss_cons.item())
        dir_meter.append(loss_dir.item() if torch.is_tensor(loss_dir) else float(loss_dir))

        writer.add_scalar('ssl/loss', loss.item(), global_step=step)
        writer.add_scalar('ssl/sup', loss_sup.item(), global_step=step)
        writer.add_scalar('ssl/cons', loss_cons.item(), global_step=step)
        writer.add_scalar('ssl/dir', dir_meter[-1], global_step=step)
        writer.add_scalar('ssl/cons_weight', cons_w, global_step=step)

        if it % config.print_interval == 0:
            log_info = (
                f'train ssl: epoch {epoch}, iter:{it}/{n_iter}, '
                f'loss: {np.mean(loss_meter):.4f}, sup: {np.mean(sup_meter):.4f}, '
                f'cons: {np.mean(cons_meter):.4f}, dir: {np.mean(dir_meter):.4f}, '
                f'lambda_cons: {cons_w:.4f}, lambda_dir: {dir_w:.4f}'
            )
            print(log_info)
            logger.info(log_info)

    scheduler.step()
    return step
