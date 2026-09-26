import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from data.dataset_camus import preprocess_grayscale_image
from data.prompts import CANONICAL
from lib import segmentation
from lib.metrics import binary_metrics


ECHONET_PROMPTS = {
    1: CANONICAL[1],
    2: CANONICAL[2],
    3: CANONICAL[3],
}


def build_tracing_masks(rows, height=112, width=112):
    from skimage.draw import polygon

    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["Frame"]), []).append(row)
    masks = {}
    for frame, points in grouped.items():
        if len(points) < 3:
            continue
        trace = points[1:]
        row_coords = [float(point["Y1"]) for point in trace]
        col_coords = [float(point["X1"]) for point in trace]
        row_coords += [float(point["Y2"]) for point in reversed(trace)]
        col_coords += [float(point["X2"]) for point in reversed(trace)]
        rr, cc = polygon(row_coords, col_coords, shape=(height, width))
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[rr, cc] = 1
        masks[frame] = mask
    return masks


def orient(array, transpose=False, flip_ud=False):
    if transpose:
        array = np.swapaxes(array, 0, 1)
    if flip_ud:
        array = np.flipud(array)
    return np.ascontiguousarray(array)


def _load_frame(video_path, frame_index, cv2):
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def evaluate_echonet(run_dir, checkpoint_name="best.pth", output_dir=None,
                     transpose=False, flip_ud=False, orientation_check=False,
                     device=None):
    import cv2
    import pandas as pd
    from PIL import Image
    from transformers import BertTokenizer

    run_dir = Path(run_dir).resolve()
    with (run_dir / "args.json").open("r", encoding="utf-8") as source:
        run_args = json.load(source)
    if run_args["text_encoder"] != "bert":
        raise ValueError("EchoNet text-prompt evaluation requires the BERT encoder")
    data_dir = Path(config.ECHONET_DATA_DIR)
    file_list = pd.read_csv(data_dir / "FileList.csv")
    traces = pd.read_csv(data_dir / "VolumeTracings.csv")
    test_cases = file_list[file_list["Split"].astype(str).str.upper() == "TEST"]
    traces_by_file = {
        str(filename): frame.to_dict("records")
        for filename, frame in traces.groupby("FileName", sort=False)
    }

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    checkpoint_path = run_dir / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = segmentation.build_model(
        swin_type=run_args["swin_type"],
        pretrained_swin="",
        window_size=run_args["window_size"],
        text_encoder="bert",
        bert_path=run_args["bert_path"],
        bert_trainable_layers=run_args["bert_trainable_layers"],
        decode_with_lang=run_args["decode_with_lang"],
        embed_tokens=run_args["embed_tokens"],
        warn_random_init=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    tokenizer = BertTokenizer.from_pretrained(run_args["bert_path"])
    prompts = {
        structure: tokenizer(
            prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=32
        )
        for structure, prompt in ECHONET_PROMPTS.items()
    }

    if output_dir is None:
        output_dir = run_dir / "eval_echonet_test"
    output_dir = Path(output_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    checked_orientation = False
    with torch.inference_mode():
        for case in test_cases.to_dict("records"):
            filename = str(case["FileName"])
            case_traces = traces_by_file.get(filename, [])
            masks = build_tracing_masks(case_traces)
            if len(masks) < 2:
                continue
            ed_frame, es_frame = sorted(masks, key=lambda frame: int(masks[frame].sum()), reverse=True)[:2]
            phases = (("ED", ed_frame), ("ES", es_frame))
            video_name = filename if filename.lower().endswith((".avi", ".mp4")) else f"{filename}.avi"
            video_path = data_dir / "Videos" / video_name
            for phase, frame_index in phases:
                rgb = _load_frame(video_path, frame_index, cv2)
                mask = masks[frame_index]
                rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA)
                rgb = orient(rgb, transpose=transpose, flip_ud=flip_ud)
                mask = orient(mask, transpose=transpose, flip_ud=flip_ud)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                image = preprocess_grayscale_image(gray, run_args["img_size"]).unsqueeze(0).to(device)

                if orientation_check and not checked_orientation:
                    from data.dataset_camus import CAMUSDataset

                    camus = CAMUSDataset(
                        config.CAMUS_DATA_DIR, split="test", img_size=112,
                        use_language=False, augment=False,
                    )
                    if camus.records:
                        camus_image = camus._load_nifti_2d(camus.records[0]["image_path"])
                        camus_image = np.clip(camus_image / max(float(camus_image.max()), 1e-6) * 255, 0, 255).astype(np.uint8)
                        camus_image = cv2.cvtColor(cv2.resize(camus_image, (112, 112)), cv2.COLOR_GRAY2RGB)
                        side_by_side = np.concatenate((camus_image, rgb), axis=1)
                        Image.fromarray(side_by_side).save(output_dir / "orientation_check.png")
                    checked_orientation = True

                frame_prompts = {}
                for structure, prompt in ECHONET_PROMPTS.items():
                    encoded = prompts[structure]
                    logits = model(
                        image,
                        encoded["input_ids"].to(device),
                        encoded["attention_mask"].to(device),
                        torch.tensor([structure - 1], device=device),
                    )
                    logits = F.interpolate(logits.float(), size=mask.shape, mode="bilinear", align_corners=False)
                    probability = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
                    prediction = probability >= 0.5
                    frame_prompts[structure] = prediction
                    if structure == 1:
                        metric = binary_metrics(prediction, mask, spacing=(1.0, 1.0))
                        rows.append({
                            "filename": filename,
                            "frame": frame_index,
                            "phase": phase,
                            "structure": "lv_endo",
                            "dice": metric["dice"],
                            "hd95_pixels": metric["hd95"],
                            "mad_pixels": metric["mad"],
                            "empty_case": metric["empty_case"],
                        })

                colors = {1: (255, 60, 60), 2: (60, 220, 100), 3: (70, 130, 255)}
                for structure, prediction in frame_prompts.items():
                    overlay = rgb.copy().astype(np.float32)
                    overlay[prediction] = 0.55 * overlay[prediction] + 0.45 * np.asarray(colors[structure])
                    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
                        overlay_dir / f"{filename}_{phase}_{structure}.png"
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "lv_metrics.csv").open("w", newline="", encoding="utf-8") as output:
        fields = ["filename", "frame", "phase", "structure", "dice", "hd95_pixels", "mad_pixels", "empty_case"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    hd95_values = np.asarray([row["hd95_pixels"] for row in rows], dtype=float)
    hd95_values = hd95_values[np.isfinite(hd95_values)]
    summary = {
        "split": "TEST",
        "quantitative_structure": "lv_endo",
        "distance_unit": "pixels",
        "n_frames": len(rows),
        "mean_dice": float(np.mean([row["dice"] for row in rows])) if rows else None,
        "mean_hd95_pixels": float(np.mean(hd95_values)) if hd95_values.size else None,
        "transpose": bool(transpose),
        "flip_ud": bool(flip_ud),
        "checkpoint": str(checkpoint_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, sort_keys=True, allow_nan=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate LV prompt segmentation on EchoNet-Dynamic test videos.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--ckpt", default="best.pth")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--flip_ud", action="store_true")
    parser.add_argument("--orientation_check", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    summary = evaluate_echonet(
        args.run_dir, args.ckpt, args.output_dir,
        transpose=args.transpose, flip_ud=args.flip_ud,
        orientation_check=args.orientation_check, device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
