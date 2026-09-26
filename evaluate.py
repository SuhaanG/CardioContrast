"""
evaluate.py — Final evaluation at native resolution (run ONCE per trained model on test).

  python evaluate.py --run_dir experiments/exp4_cardiocontrast_seed42
  python evaluate.py --run_dir ... --prompt_set heldout      # language generalisation
  python evaluate.py --run_dir ... --split val              # debugging only

Writes to <run_dir>/eval_<split>_<prompt_set>/:
  per_sample.csv   one row per (image, structure, prompt): dice, iou, hd95, hd,
                   mad (mm), leakage, patient/view/phase/quality
  per_patient.csv  per-patient means (input for compare_runs.py)
  clinical.csv     per-patient EDV/ESV/EF from predicted and GT masks
  summary.json     per structure x phase: mean/std/median of each metric,
                   failure counts, leakage, EF/EDV/ESV agreement, per image-
                   quality breakdown, and a sanity check of the Simpson code
                   (GT-mask EF vs the EF in the CAMUS cfg files).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import config
from data.dataset_camus import CAMUSDataset
from lib.report import Reporter
from lib.segmentation import build_model


def load_run(run_dir, ckpt_name, device):
    ck = torch.load(os.path.join(run_dir, ckpt_name), map_location="cpu", weights_only=False)
    a = ck["args"]
    model = build_model(a["swin_type"], "", a["window_size"], a["text_encoder"],
                        a["bert_path"], a["bert_trainable_layers"], a["decode_with_lang"],
                        a.get("embed_tokens", 8))
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    return model, a


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpt", default="best.pth")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--prompt_set", default="canonical", choices=["canonical", "heldout"])
    p.add_argument("--data_dir", default=config.CAMUS_DATA_DIR)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_preds", action="store_true", help="save native-res masks (npz)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_images", type=int, default=None)
    args = p.parse_args()

    model, a = load_run(args.run_dir, args.ckpt, args.device)
    if args.prompt_set == "heldout" and a["text_encoder"] == "embedding":
        raise SystemExit("Held-out prompts are meaningless for the class-embedding model.")

    ds = CAMUSDataset(args.data_dir, args.split, a["img_size"], a["bert_path"], augment=False,
                      eval_prompt_set=args.prompt_set, seed=a["seed"],
                      max_images=args.max_images)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    out_dir = os.path.join(args.run_dir, "eval_{}_{}".format(args.split, args.prompt_set))
    pred_dir = os.path.join(out_dir, "preds")
    if args.save_preds:
        os.makedirs(pred_dir, exist_ok=True)

    use_amp = args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
    rep = Reporter()
    gt_cache = {}

    for batch in loader:
        img = batch["image"].to(args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(img, batch["input_ids"].to(args.device),
                           batch["attn_mask"].to(args.device), batch["class_id"].to(args.device))
        logits = logits.float()
        for j in range(img.shape[0]):
            image_idx = int(batch["image_idx"][j])
            s = int(batch["structure"][j])
            h, w = batch["orig_size"][j].tolist()
            if image_idx not in gt_cache:
                if len(gt_cache) > 64:
                    gt_cache.clear()
                gt_cache[image_idx] = ds.load_full_res(image_idx)
            gt_full, spacing = gt_cache[image_idx]
            # Upsample LOGITS to native size, then threshold (not the other way round).
            prob = torch.softmax(F.interpolate(logits[j:j + 1], size=(h, w), mode="bilinear",
                                               align_corners=False), dim=1)[0, 1].cpu().numpy()
            pred = prob > 0.5
            rec = ds.images[image_idx]
            rep.add(rec["patient"], rec["view"], rec["phase"], rec["quality"], rec["ref_ef"],
                    s, batch["prompt"][j], pred, gt_full, spacing)
            if args.save_preds:
                np.savez_compressed(
                    os.path.join(pred_dir, "{}_{}_{}_s{}_{}.npz".format(
                        rec["patient"], rec["view"], rec["phase"], s, len(rep.rows))),
                    pred=pred.astype(np.uint8), prob=prob.astype(np.float16))

    print("\n=== {} / {} / {} ===".format(os.path.basename(os.path.normpath(args.run_dir)),
                                         args.split, args.prompt_set))
    rep.write(out_dir, meta={"run_dir": args.run_dir, "checkpoint": args.ckpt,
                             "split": args.split, "prompt_set": args.prompt_set,
                             "train_args": a})


if __name__ == "__main__":
    main()
