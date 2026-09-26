"""
evaluate_echonet.py — Out-of-distribution test on EchoNet-Dynamic (never trained on).

What EchoNet-Dynamic can and cannot test
----------------------------------------
EchoNet-Dynamic provides LV endocardial tracings at two frames per video
(end-diastole / end-systole) at 112x112, and NO myocardium or left-atrium
labels. So this script gives QUANTITATIVE results for the LV prompt only.
Myocardium / LA predictions can be saved for QUALITATIVE figures, but the paper
must not claim quantitative OOD disambiguation from EchoNet.
Pixel spacing is not provided, so distances are reported in pixels (112x112).

Step 0 (mandatory): check orientation. CAMUS NIfTI arrays and EchoNet frames may
not share orientation. Run
    python evaluate_echonet.py --run_dir <run> --orientation_check
look at orientation_check.png, and pass --transpose / --flip_ud if the EchoNet
frame is not oriented like the CAMUS image the model was trained on.

    python evaluate_echonet.py --run_dir experiments/exp4_cardiocontrast_seed42 [--transpose]
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.draw import polygon

import config
from data.dataset_camus import IMAGENET_MEAN, IMAGENET_STD, index_images, load_nifti_2d
from data.prompts import CANONICAL
from evaluate import load_run
from lib.metrics import binary_metrics


def load_tracings(root):
    tr = pd.read_csv(os.path.join(root, "VolumeTracings.csv"))
    tr["FileName"] = tr["FileName"].astype(str)
    out = {}
    for (fname, frame), g in tr.groupby(["FileName", "Frame"], sort=False):
        out.setdefault(fname, {})[int(frame)] = g[["X1", "Y1", "X2", "Y2"]].to_numpy(float)
    return out


def trace_to_mask(t, size=112):
    x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    x = np.concatenate((x1[1:], np.flip(x2[1:])))
    y = np.concatenate((y1[1:], np.flip(y2[1:])))
    rr, cc = polygon(np.rint(y).astype(int), np.rint(x).astype(int), (size, size))
    m = np.zeros((size, size), bool)
    m[rr, cc] = True
    return m


def read_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)


def orient(img, transpose, flip_ud):
    if transpose:
        img = img.T
    if flip_ud:
        img = img[::-1]
    return np.ascontiguousarray(img)


def preprocess(gray, img_size):
    x = gray.astype(np.float32)
    x = x / x.max() if x.max() > 0 else x
    t = torch.from_numpy(x)[None, None]
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False,
                      antialias=True)[0]
    return (t.repeat(3, 1, 1) - IMAGENET_MEAN) / IMAGENET_STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="best.pth")
    ap.add_argument("--echonet_dir", default=config.ECHONET_DATA_DIR)
    ap.add_argument("--camus_dir", default=config.CAMUS_DATA_DIR)
    ap.add_argument("--split", default="TEST")
    ap.add_argument("--transpose", action="store_true")
    ap.add_argument("--flip_ud", action="store_true")
    ap.add_argument("--orientation_check", action="store_true")
    ap.add_argument("--save_examples", type=int, default=20,
                    help="save overlays (LV/MYO/LA prompts) for N videos")
    ap.add_argument("--max_videos", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, a = load_run(args.run_dir, args.ckpt, args.device)
    out_dir = os.path.join(args.run_dir, "eval_echonet_{}".format(args.split.lower()))
    os.makedirs(out_dir, exist_ok=True)

    fl = pd.read_csv(os.path.join(args.echonet_dir, "FileList.csv"))
    fl = fl[fl["Split"].str.upper() == args.split.upper()]
    names = [n if str(n).endswith(".avi") else str(n) + ".avi" for n in fl["FileName"]]
    if args.max_videos:
        names = names[: args.max_videos]
    traces = load_tracings(args.echonet_dir)

    if args.orientation_check:
        cam = index_images(args.camus_dir, "train", verbose=False)[0]
        cimg = load_nifti_2d(cam["image_path"])[0].astype(np.float32)
        cimg = (cimg / cimg.max() * 255).astype(np.uint8)
        e = read_frame(os.path.join(args.echonet_dir, "Videos", names[0]), 0)
        e = orient(e, args.transpose, args.flip_ud)
        a1 = Image.fromarray(cimg).resize((256, 256))
        a2 = Image.fromarray(e).resize((256, 256))
        canvas = Image.new("L", (520, 256))
        canvas.paste(a1, (0, 0))
        canvas.paste(a2, (264, 0))
        p = os.path.join(out_dir, "orientation_check.png")
        canvas.save(p)
        print("Left: CAMUS as the model sees it. Right: EchoNet with current flags. Saved", p)
        return

    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(a["bert_path"])

    def encode(prompt):
        ids = tok.encode(prompt, add_special_tokens=True)[:20]
        i = torch.zeros(1, 20, dtype=torch.long)
        m = torch.zeros(1, 20, dtype=torch.long)
        i[0, :len(ids)] = torch.tensor(ids)
        m[0, :len(ids)] = 1
        return i.to(args.device), m.to(args.device)

    prompts = {s: encode(CANONICAL[s]) for s in (1, 2, 3)}
    rows = []
    ex_dir = os.path.join(out_dir, "examples")
    os.makedirs(ex_dir, exist_ok=True)

    for vi, name in enumerate(names):
        if name not in traces:
            continue
        frames = traces[name]
        masks = {f: trace_to_mask(t) for f, t in frames.items()}
        if len(masks) < 2:
            continue
        # Larger traced area = end-diastole.
        order = sorted(masks, key=lambda f: masks[f].sum(), reverse=True)
        for phase, f in (("ED", order[0]), ("ES", order[-1])):
            gray = read_frame(os.path.join(args.echonet_dir, "Videos", name), f)
            if gray is None:
                continue
            gt = orient(masks[f].astype(np.uint8), args.transpose, args.flip_ud).astype(bool)
            g = orient(gray, args.transpose, args.flip_ud)
            x = preprocess(g, a["img_size"])[None].to(args.device)
            preds = {}
            with torch.no_grad():
                for s in (1, 2, 3):
                    ids, am = prompts[s]
                    lg = model(x, ids, am, torch.tensor([s - 1], device=args.device)).float()
                    lg = F.interpolate(lg, size=g.shape, mode="bilinear", align_corners=False)
                    preds[s] = (lg.argmax(1)[0].cpu().numpy() == 1)
            m = binary_metrics(preds[1], gt, (1.0, 1.0))
            rows.append({"video": name, "phase": phase, "frame": f, **m,
                         "lv_pred_overlaps_myo_pred": float((preds[1] & preds[2]).sum() /
                                                           max(1, preds[1].sum()))})
            if vi < args.save_examples:
                rgb = np.stack([g] * 3, -1).astype(np.float32)
                for s, col in ((1, (255, 0, 0)), (2, (0, 255, 0)), (3, (0, 0, 255))):
                    rgb[preds[s]] = 0.55 * rgb[preds[s]] + 0.45 * np.array(col)
                edge = gt ^ cv2.erode(gt.astype(np.uint8), np.ones((3, 3))).astype(bool)
                rgb[edge] = (255, 255, 0)
                Image.fromarray(rgb.astype(np.uint8)).resize((336, 336), Image.NEAREST).save(
                    os.path.join(ex_dir, "{}_{}.png".format(name[:-4], phase)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "per_frame.csv"), index=False)
    print("\nEchoNet-Dynamic {} ({} frames, {} videos) — LV prompt only".format(
        args.split, len(df), df["video"].nunique() if len(df) else 0))
    for ph, gp in list(df.groupby("phase")) + [("all", df)]:
        print("  {:3s} Dice {:.4f}±{:.4f}  HD95 {:.2f}px  empty preds {}".format(
            ph, gp["dice"].mean(), gp["dice"].std(), np.nanmean(gp["hd95"]),
            int((gp["empty_case"] == "pred").sum())))
    print("Overlays (red=LV, green=MYO, blue=LA, yellow=GT LV contour) in", ex_dir)


if __name__ == "__main__":
    main()
