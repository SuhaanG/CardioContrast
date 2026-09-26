import math

import numpy as np
from scipy import ndimage


def _surface_from_mask(mask, kernel_size=3):
    mask = np.asarray(mask) > 0
    if mask.size == 0:
        return np.zeros_like(mask, dtype=np.bool_)
    if mask.ndim != 2:
        mask = mask.squeeze()
    eroded = ndimage.binary_erosion(mask, structure=np.ones((kernel_size, kernel_size), dtype=bool), iterations=1)
    return mask & ~eroded


def binary_metrics(pred, gt, spacing):
    pred = np.asarray(pred, dtype=np.uint8)
    gt = np.asarray(gt, dtype=np.uint8)
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
        dists = np.concatenate((dist_pred[gt_surface], dist_gt[pred_surface])).astype(float, copy=False)
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


def _long_axis_diameters(lv_mask, la_mask, spacing, n_discs):
    lv = np.asarray(lv_mask).squeeze() > 0
    if lv.ndim != 2 or not lv.any():
        return None, 0.0, max(spacing)
    sx, sy = map(float, spacing)
    if sx <= 0 or sy <= 0:
        raise ValueError("Pixel spacing must be positive")

    coordinates = np.argwhere(lv).astype(float)
    physical = coordinates * np.asarray((sy, sx))
    surface = _surface_from_mask(lv)
    la = None if la_mask is None else np.asarray(la_mask).squeeze() > 0
    if la is not None and la.shape != lv.shape:
        raise ValueError("LV and LA masks for a view must have identical shapes")

    contact = np.empty((0, 2), dtype=float)
    if la is not None and la.any():
        adjacent_la = ndimage.binary_dilation(la, structure=np.ones((3, 3), dtype=bool))
        contact_pixels = np.argwhere(surface & adjacent_la)
        if contact_pixels.size:
            contact = contact_pixels.astype(float) * np.asarray((sy, sx))

    if contact.size:
        base = contact.mean(axis=0)
    else:
        centered = physical - physical.mean(axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        principal_axis = axes[0]
        projections = centered @ principal_axis
        endpoints = np.stack((physical[np.argmin(projections)], physical[np.argmax(projections)]))
        if la is not None and la.any():
            la_center = np.argwhere(la).mean(axis=0) * np.asarray((sy, sx))
            base = endpoints[np.argmin(np.linalg.norm(endpoints - la_center, axis=1))]
        else:
            base = endpoints[0]

    apex = physical[np.argmax(np.linalg.norm(physical - base, axis=1))]
    axis = apex - base
    length = float(np.linalg.norm(axis))
    if length <= 0:
        return None, 0.0, max(spacing)
    axis /= length
    perpendicular = np.asarray((-axis[1], axis[0]))
    along = (physical - base) @ axis
    across = (physical - base) @ perpendicular
    pixel_step_along = float(np.hypot(axis[0] * sy, axis[1] * sx))
    pixel_step_across = float(np.hypot(perpendicular[0] * sy, perpendicular[1] * sx))
    half_band = 0.75 * pixel_step_along

    diameters = []
    for disc in range(n_discs):
        center = (disc + 0.5) * length / n_discs
        selected = np.abs(along - center) <= half_band
        if not selected.any():
            selected[np.argmin(np.abs(along - center))] = True
        diameter = float(across[selected].max() - across[selected].min() + pixel_step_across)
        diameters.append(diameter)
    return np.asarray(diameters), length, max(sx, sy)


def simpson_volume(lv_2ch, lv_4ch, la_2ch=None, la_4ch=None,
                   spacing_2ch=(1.0, 1.0), spacing_4ch=(1.0, 1.0), n_discs=20):
    """Estimate biplane LV volume in mL from ED or ES 2CH/4CH masks.

    Spacing tuples are ordered as (x, y) in millimeters. LA masks are optional;
    without them the long-axis base is chosen from the principal-axis extremes.
    """
    if n_discs < 1:
        raise ValueError("n_discs must be positive")
    if not np.asarray(lv_2ch).size or not np.asarray(lv_4ch).size:
        return 0.0
    diameters_2ch, length_2ch, pixel_2ch = _long_axis_diameters(
        lv_2ch, la_2ch, spacing_2ch, n_discs
    )
    diameters_4ch, length_4ch, pixel_4ch = _long_axis_diameters(
        lv_4ch, la_4ch, spacing_4ch, n_discs
    )
    if diameters_2ch is None or diameters_4ch is None:
        return 0.0
    length = max(length_2ch, length_4ch) + max(pixel_2ch, pixel_4ch)
    volume_mm3 = (math.pi / 4.0) * np.sum(diameters_2ch * diameters_4ch) * length / n_discs
    return float(volume_mm3 / 1000.0)


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


def bootstrap_confidence_interval(values, confidence=0.95, n_bootstrap=2000, seed=42):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"lower": np.nan, "estimate": np.nan, "upper": np.nan, "n": 0}

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(n_bootstrap)):
        sample = rng.choice(values, size=values.size, replace=True)
        boot.append(float(np.mean(sample)))
    boot = np.asarray(boot, dtype=float)
    alpha = 1.0 - confidence
    lower = np.quantile(boot, alpha / 2.0)
    upper = np.quantile(boot, 1.0 - alpha / 2.0)
    return {
        "lower": float(lower),
        "estimate": float(np.mean(values)),
        "upper": float(upper),
        "n": int(values.size),
        "confidence": float(confidence),
    }


__all__ = [
    "binary_metrics",
    "leakage",
    "simpson_volume",
    "ejection_fraction",
    "agreement",
    "per_structure_metrics",
    "bootstrap_confidence_interval",
]
