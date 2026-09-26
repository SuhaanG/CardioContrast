import math

import numpy as np
from scipy import ndimage


def _surface_from_mask(mask, kernel_size=3):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.size == 0:
        return np.zeros_like(mask, dtype=np.bool_)
    if mask.ndim != 2:
        mask = mask.squeeze()
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((kernel_size, kernel_size), dtype=bool), iterations=1)
    return mask & ~eroded


def binary_metrics(pred, gt, spacing):
    pred = np.asarray(pred, dtype=np.uint8)
    gt = np.asarray(gt, dtype=np.uint8)
    empty_case = bool((pred.sum() == 0) or (gt.sum() == 0))
    if pred.size == 0 and gt.size == 0:
        return {"dice": 1.0, "iou": 1.0, "hd95": np.nan, "hd": np.nan, "mad": np.nan, "empty_case": True}
    if pred.sum() == 0 and gt.sum() == 0:
        return {"dice": 1.0, "iou": 1.0, "hd95": np.nan, "hd": np.nan, "mad": np.nan, "empty_case": True}
    if pred.sum() == 0 or gt.sum() == 0:
        return {"dice": 0.0, "iou": 0.0, "hd95": np.nan, "hd": np.nan, "mad": np.nan, "empty_case": True}

    inter = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    dice = 2.0 * inter / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0
    iou = inter / union if union > 0 else 0.0

    pred_surface = _surface_from_mask(pred)
    gt_surface = _surface_from_mask(gt)
    if pred_surface.any() and gt_surface.any():
        dist_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
        dist_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
        distances = np.array([dist_pred[gt_surface], dist_gt[pred_surface]], dtype=float)
        dists = np.concatenate([distances[0], distances[1]])
        hd = float(np.max(dists)) if dists.size else np.nan
        hd95 = float(np.percentile(dists, 95)) if dists.size else np.nan
        mad = float(np.mean(dists)) if dists.size else np.nan
    else:
        hd95 = hd = mad = np.nan
    return {"dice": float(dice), "iou": float(iou), "hd95": float(hd95), "hd": float(hd), "mad": float(mad), "empty_case": False}


def leakage(pred, multilabel_gt, structure):
    pred = np.asarray(pred, dtype=np.uint8)
    gt = np.asarray(multilabel_gt, dtype=np.uint8)
    other = (gt > 0) & (gt != structure)
    if pred.sum() == 0:
        return 0.0
    false_inside = np.logical_and(pred > 0, other)
    return float(false_inside.sum() / max(1, (pred > 0).sum()))


def simpson_volume(mask, _ignored=None, spacing_x=1.0, spacing_y=1.0):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        mask = mask.squeeze()
    if mask.size == 0:
        return 0.0
    area = mask.sum() * spacing_x * spacing_y
    return float(area)


def ejection_fraction(edv, esv):
    return 100.0 * (edv - esv) / edv if edv > 0 else 0.0


def agreement(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = int(len(x))
    if n < 2:
        return {"n": n, "r": np.nan, "bias": np.nan, "mae": np.nan, "loa": (np.nan, np.nan)}
    r = np.corrcoef(x, y)[0, 1]
    bias = float(np.mean(x - y))
    mae = float(np.mean(np.abs(x - y)))
    diff = x - y
    loa = (float(np.mean(diff) - 1.96 * np.std(diff, ddof=1)), float(np.mean(diff) + 1.96 * np.std(diff, ddof=1)))
    return {"n": n, "r": float(r), "bias": bias, "mae": mae, "loa": loa}


def per_structure_metrics(pred, gt, structure_id=1, spacing=(1.0, 1.0)):
    pred = np.asarray(pred, dtype=np.uint8)
    gt = np.asarray(gt, dtype=np.uint8)
    pred_area = (pred > 0).sum()
    gt_area = (gt > 0).sum()
    empty_case = bool((pred_area == 0) or (gt_area == 0))

    if empty_case:
        return {
            "structure_id": int(structure_id),
            "dice": 1.0 if pred_area == 0 and gt_area == 0 else 0.0,
            "iou": 1.0 if pred_area == 0 and gt_area == 0 else 0.0,
            "hd95": np.nan,
            "hd": np.nan,
            "mad": np.nan,
            "empty_case": empty_case,
        }

    base = binary_metrics(pred, gt, spacing=spacing)
    return {
        "structure_id": int(structure_id),
        "dice": float(base["dice"]),
        "iou": float(base["iou"]),
        "hd95": float(base["hd95"]),
        "hd": float(base["hd"]),
        "mad": float(base["mad"]),
        "empty_case": False,
    }


__all__ = ["binary_metrics", "leakage", "simpson_volume", "ejection_fraction", "agreement", "per_structure_metrics"]
