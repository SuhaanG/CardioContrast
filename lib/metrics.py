"""
lib/metrics.py — Segmentation and clinical metrics used by CAMUS/TMI papers.

All geometric metrics are computed at NATIVE image resolution with the pixel
spacing from the NIfTI header, so distances are in millimetres.

  dice, iou                       overlap
  hd95, hd                        95th-percentile / max symmetric Hausdorff (mm)
  mad                             mean absolute (symmetric surface) distance (mm)
  leakage                         fraction of predicted pixels that fall inside a
                                  DIFFERENT structure's ground truth -> direct
                                  measure of cross-structure confusion, i.e. the
                                  "disambiguation" the paper claims to improve
  simpson_biplane_volume / ef     LV volumes (mL) and ejection fraction (%) by the
                                  biplane method of discs from 2CH + 4CH masks

Empty-mask conventions (report counts of these in the paper):
  GT empty & pred empty -> dice = iou = 1, distances = NaN
  exactly one empty     -> dice = iou = 0, distances = NaN (counted as failures)
"""

import numpy as np
from scipy import ndimage


def _surface(mask):
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)
    return mask & ~eroded


def surface_distances(pred, gt, spacing):
    """Distances (mm) from each surface pixel of pred to gt surface and vice versa."""
    ps, gs = _surface(pred), _surface(gt)
    dt_g = ndimage.distance_transform_edt(~gs, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~ps, sampling=spacing)
    return dt_g[ps], dt_p[gs]


def binary_metrics(pred, gt, spacing):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    psum, gsum = pred.sum(), gt.sum()
    union = psum + gsum - inter
    out = {}
    if psum == 0 and gsum == 0:
        out.update(dice=1.0, iou=1.0, hd95=np.nan, hd=np.nan, mad=np.nan, empty_case="both")
        return out
    out["dice"] = 2.0 * inter / (psum + gsum)
    out["iou"] = inter / union
    if psum == 0 or gsum == 0:
        out.update(hd95=np.nan, hd=np.nan, mad=np.nan,
                   empty_case="pred" if psum == 0 else "gt")
        return out
    d_pg, d_gp = surface_distances(pred, gt, spacing)
    both = np.concatenate([d_pg, d_gp])
    out["hd95"] = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    out["hd"] = float(max(d_pg.max(), d_gp.max()))
    out["mad"] = float(both.mean())
    out["empty_case"] = "none"
    return out


def leakage(pred, multilabel_gt, structure):
    """Fraction of predicted pixels lying in the GT of any OTHER structure."""
    pred = pred.astype(bool)
    n = pred.sum()
    if n == 0:
        return np.nan
    other = (multilabel_gt > 0) & (multilabel_gt != structure)
    return float(np.logical_and(pred, other).sum() / n)


# --------------------------------------------------------------------------- EF
def _pts_mm(mask, spacing):
    return np.argwhere(mask).astype(np.float64) * np.asarray(spacing, dtype=np.float64)


def lv_long_axis(lv, spacing, la=None):
    """
    Return (base_midpoint, apex) in mm.
    Base = midpoint of the LV/LA contact (mitral annulus proxy) when an LA mask is
    given and touches the LV; apex = LV pixel farthest from that midpoint.
    Fallback: the two extremes of the LV along its principal axis.
    """
    pts = _pts_mm(lv, spacing)
    if la is not None and la.any():
        contact = lv & ndimage.binary_dilation(la, iterations=2)
        if contact.sum() >= 2:
            c = _pts_mm(contact, spacing)
            cc = c - c.mean(0)
            u = np.linalg.svd(cc, full_matrices=False)[2][0]
            proj = cc @ u
            base = (c[proj.argmin()] + c[proj.argmax()]) / 2.0
            apex = pts[np.linalg.norm(pts - base, axis=1).argmax()]
            return base, apex
    cen = pts.mean(0)
    u = np.linalg.svd(pts - cen, full_matrices=False)[2][0]
    proj = (pts - cen) @ u
    return pts[proj.argmin()], pts[proj.argmax()]


def disc_diameters(lv, spacing, la=None, n_discs=20):
    """Diameters (mm) of n_discs slabs perpendicular to the long axis, and length L."""
    if lv.sum() < 10:
        return np.zeros(n_discs), 0.0
    base, apex = lv_long_axis(lv, spacing, la)
    axis = apex - base
    L = float(np.linalg.norm(axis))
    if L == 0:
        return np.zeros(n_discs), 0.0
    u = axis / L
    perp = np.array([-u[1], u[0]])
    pts = _pts_mm(lv, spacing)
    t = (pts - base) @ u
    s = (pts - base) @ perp
    px = float(np.mean(spacing))
    diam = np.zeros(n_discs)
    edges = np.linspace(0, L, n_discs + 1)
    for i in range(n_discs):
        # Diameter measured at the disc MID-plane (thin band), as in the method
        # of discs; the max over the whole slab would over-estimate volume.
        mid = 0.5 * (edges[i] + edges[i + 1])
        sel = np.abs(t - mid) <= 0.75 * px
        if not sel.any():
            sel = (t >= edges[i]) & (t < edges[i + 1])
        if sel.any():
            diam[i] = s[sel].max() - s[sel].min() + px
    return diam, L + px   # + px: base/apex are pixel centres


def simpson_biplane_volume(lv_2ch, sp_2ch, lv_4ch, sp_4ch, la_2ch=None, la_4ch=None,
                           n_discs=20):
    """LV volume in mL by the biplane method of discs."""
    a, L2 = disc_diameters(lv_2ch.astype(bool), sp_2ch,
                           None if la_2ch is None else la_2ch.astype(bool), n_discs)
    b, L4 = disc_diameters(lv_4ch.astype(bool), sp_4ch,
                           None if la_4ch is None else la_4ch.astype(bool), n_discs)
    L = max(L2, L4)
    vol_mm3 = np.pi / 4.0 * np.sum(a * b) * (L / n_discs)
    return vol_mm3 / 1000.0


def ejection_fraction(edv, esv):
    if edv <= 0:
        return np.nan
    return 100.0 * (edv - esv) / edv


def agreement(x, y):
    """Pearson r, bias (mean y-x), MAE, 95% limits of agreement."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return {"n": int(len(x))}
    d = y - x
    return {
        "n": int(len(x)),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "bias": float(d.mean()),
        "mae": float(np.abs(d).mean()),
        "loa_low": float(d.mean() - 1.96 * d.std(ddof=1)),
        "loa_high": float(d.mean() + 1.96 * d.std(ddof=1)),
    }
