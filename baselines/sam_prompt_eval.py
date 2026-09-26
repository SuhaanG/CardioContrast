"""
baselines/sam_prompt_eval.py — Geometric-prompt baselines (SAM / MedSAM).

The paper's motivation is that geometric prompts are ambiguous for co-located
cardiac structures. Right now that is asserted, not shown. This script
measures it on the same test images with the same metrics (lib/report.py),
including the `leakage` metric (fraction of the predicted mask inside another
structure), which is where the ambiguity should show up -- e.g. a myocardium
box also encloses the LV cavity.

Prompt types (all derived from GT, i.e. an optimistic "oracle user"):
  box             GT bounding box, each side jittered by up to --box_jitter x size
  point_center    one positive click at the most interior pixel of the structure
  point_boundary  one positive click inside the structure but within
                  --boundary_mm of a DIFFERENT structure (the ambiguous case)

Install:  pip install git+https://github.com/facebookresearch/segment-anything.git
Weights:  SAM   sam_vit_b_01ec64.pth (or vit_h) from the SAM repo
          MedSAM medsam_vit_b.pth from the MedSAM repo (box prompts only)

  python baselines/sam_prompt_eval.py --model sam    --ckpt sam_vit_b_01ec64.pth --prompt box
  python baselines/sam_prompt_eval.py --model sam    --ckpt sam_vit_b_01ec64.pth --prompt point_boundary
  python baselines/sam_prompt_eval.py --model medsam --ckpt medsam_vit_b.pth     --prompt box

The MedSAM path follows the public MedSAM inference recipe (1024x1024 resize,
min-max normalisation, single-mask output). Verify against the version of the
MedSAM repo you download before reporting numbers.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from data.dataset_camus import index_images, load_nifti_2d  # noqa: E402
from lib.report import Reporter  # noqa: E402


def bbox(mask):
    ys, xs = np.where(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)  # x0,y0,x1,y1


def jitter_box(box, frac, rng, shape):
    w, h = box[2] - box[0], box[3] - box[1]
    d = rng.uniform(-frac, frac, size=4) * np.array([w, h, w, h])
    b = box + d
    b[[0, 2]] = np.clip(b[[0, 2]], 0, shape[1] - 1)
    b[[1, 3]] = np.clip(b[[1, 3]], 0, shape[0] - 1)
    return b


def center_point(mask):
    dt = ndimage.distance_transform_edt(mask)
    y, x = np.unravel_index(dt.argmax(), dt.shape)
    return np.array([[x, y]], dtype=np.float32)


def boundary_point(mask, gt, s, spacing, boundary_mm, rng):
    other = (gt > 0) & (gt != s)
    if not other.any():
        return center_point(mask)
    dist_other = ndimage.distance_transform_edt(~other, sampling=spacing)
    cand = mask & (dist_other <= boundary_mm)
    # stay at least 1 px inside the structure
    cand &= ndimage.binary_erosion(mask)
    if not cand.any():
        return center_point(mask)
    ys, xs = np.where(cand)
    k = rng.integers(len(ys))
    return np.array([[xs[k], ys[k]]], dtype=np.float32)


class SAMRunner:
    def __init__(self, ckpt, arch, device):
        from segment_anything import SamPredictor, sam_model_registry
        self.p = SamPredictor(sam_model_registry[arch](checkpoint=ckpt).to(device))

    def set_image(self, rgb):
        self.p.set_image(rgb)

    def predict(self, box=None, point=None):
        if box is not None:
            m, sc, _ = self.p.predict(box=box, multimask_output=False)
            return m[0]
        m, sc, _ = self.p.predict(point_coords=point, point_labels=np.array([1]),
                                  multimask_output=True)
        return m[int(np.argmax(sc))]


class MedSAMRunner:
    def __init__(self, ckpt, arch, device):
        from segment_anything import sam_model_registry
        self.m = sam_model_registry[arch](checkpoint=ckpt).to(device).eval()
        self.device = device

    @torch.no_grad()
    def set_image(self, rgb):
        self.H, self.W = rgb.shape[:2]
        x = torch.from_numpy(rgb).float().permute(2, 0, 1)[None]
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        x = (x - x.min()) / (x.max() - x.min()).clamp(min=1e-8)
        self.emb = self.m.image_encoder(x.to(self.device))

    @torch.no_grad()
    def predict(self, box=None, point=None):
        assert box is not None, "MedSAM is used with box prompts only"
        b = box / np.array([self.W, self.H, self.W, self.H]) * 1024
        bt = torch.as_tensor(b, dtype=torch.float, device=self.device)[None, None]
        sp, de = self.m.prompt_encoder(points=None, boxes=bt, masks=None)
        logits, _ = self.m.mask_decoder(image_embeddings=self.emb,
                                        image_pe=self.m.prompt_encoder.get_dense_pe(),
                                        sparse_prompt_embeddings=sp, dense_prompt_embeddings=de,
                                        multimask_output=False)
        prob = torch.sigmoid(F.interpolate(logits, size=(self.H, self.W), mode="bilinear",
                                           align_corners=False))
        return prob[0, 0].cpu().numpy() > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sam", "medsam"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="vit_b")
    ap.add_argument("--prompt", choices=["box", "point_center", "point_boundary"], default="box")
    ap.add_argument("--box_jitter", type=float, default=0.05)
    ap.add_argument("--boundary_mm", type=float, default=3.0)
    ap.add_argument("--data_dir", default=config.CAMUS_DATA_DIR)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_root", default=os.path.join(config.OUTPUT_ROOT, "baselines"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.model == "medsam" and args.prompt != "box":
        raise SystemExit("MedSAM is evaluated with box prompts only.")

    runner = (SAMRunner if args.model == "sam" else MedSAMRunner)(args.ckpt, args.arch,
                                                                   args.device)
    rng = np.random.default_rng(args.seed)
    rep = Reporter()
    for r in index_images(args.data_dir, args.split):
        img, spacing = load_nifti_2d(r["image_path"])
        gt, _ = load_nifti_2d(r["mask_path"])
        gt = gt.astype(np.uint8)
        img = img.astype(np.float32)
        img = (img / max(img.max(), 1e-8) * 255).astype(np.uint8)
        runner.set_image(np.stack([img] * 3, -1))
        for s in (1, 2, 3):
            m = gt == s
            if not m.any():
                continue
            if args.prompt == "box":
                pred = runner.predict(box=jitter_box(bbox(m), args.box_jitter, rng, m.shape))
            elif args.prompt == "point_center":
                pred = runner.predict(point=center_point(m))
            else:
                pred = runner.predict(point=boundary_point(m, gt, s, spacing, args.boundary_mm,
                                                           rng))
            rep.add(r["patient"], r["view"], r["phase"], r["quality"], r["ref_ef"], s,
                    args.prompt, pred, gt, spacing)

    name = "{}_{}_{}".format(args.model, args.arch, args.prompt)
    rep.write(os.path.join(args.out_root, name, "eval_{}_canonical".format(args.split)),
              meta={"model": name, "ckpt": args.ckpt, "prompt": args.prompt,
                    "box_jitter": args.box_jitter, "boundary_mm": args.boundary_mm})


if __name__ == "__main__":
    main()
