"""Binary metrics at the model's resized grid; HD95 is in pixels, not mm."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def overlap(tp, fp, fn):
    # Empty/empty = 1; one empty = 0. Counts also support pooled aggregation.
    return dict(dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
                iou=tp / (tp + fp + fn) if tp + fp + fn else 1.0)


def binary_metrics(pred, target, tolerance=2):
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    if pred.ndim != 2 or pred.shape != target.shape or tolerance < 0:
        raise ValueError('Expected equal 2D masks and nonnegative tolerance')
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    result = dict(tp=tp, fp=fp, fn=fn, **overlap(tp, fp, fn))
    fraction = float(target.mean())
    result.update(foreground_fraction=fraction,
                  size_group='small' if fraction < 0.05 else ('medium' if fraction < 0.2 else 'large'))
    if not pred.any() or not target.any():
        both_empty = not pred.any() and not target.any()
        result.update(boundary_f1=1.0 if both_empty else 0.0,
                      hd95_px=0.0 if both_empty else None,
                      empty_case='both' if both_empty else ('prediction' if not pred.any() else 'target'))
        return result
    pb = pred ^ binary_erosion(pred, border_value=0)
    tb = target ^ binary_erosion(target, border_value=0)
    pred_dist = distance_transform_edt(~tb)[pb]
    target_dist = distance_transform_edt(~pb)[tb]
    precision = float((pred_dist <= tolerance).mean())
    recall = float((target_dist <= tolerance).mean())
    result.update(boundary_f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                  hd95_px=float(np.percentile(np.concatenate([pred_dist, target_dist]), 95)),
                  empty_case='none')
    return result


def summarize(rows):
    if not rows:
        raise ValueError('No evaluation samples')
    pooled = overlap(*(sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn')))
    hd = [r['hd95_px'] for r in rows if r['hd95_px'] is not None]
    result = dict(n=len(rows), pooled_dice=pooled['dice'], pooled_iou=pooled['iou'],
                  macro_dice=float(np.mean([r['dice'] for r in rows])),
                  macro_iou=float(np.mean([r['iou'] for r in rows])),
                  boundary_f1=float(np.mean([r['boundary_f1'] for r in rows])),
                  hd95_px_valid_mean=float(np.mean(hd)) if hd else None,
                  hd95_undefined_count=len(rows) - len(hd))
    result['size_groups'] = {
        group: dict(n=len(items), macro_dice=float(np.mean([r['dice'] for r in items])),
                    boundary_f1=float(np.mean([r['boundary_f1'] for r in items])))
        for group in ('small', 'medium', 'large')
        for items in [[r for r in rows if r['size_group'] == group]] if items
    }
    return result
