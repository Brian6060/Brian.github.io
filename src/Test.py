#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import torch

from Model import UNetPaper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure inference for paper-style U-Net.")

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0/processed_unet_test"),
        help="Processed dataset root.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="test_images.json",
        help="Manifest filename under processed_root/manifests/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0/outputs_train_formal_A/best_train_loss.pt"),
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0/outputs_test_formal_A_best"),
        help="Directory to save inference outputs.",
    )

    # 必须和训练一致
    parser.add_argument("--input-size", type=int, default=572)
    parser.add_argument("--output-size", type=int, default=388)
    parser.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "minmax", "none"])

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    parser.add_argument("--save-pred-tif", action="store_true", default=True)
    parser.add_argument("--save-pred-npy", action="store_true", default=True)
    parser.add_argument("--save-prob-npy", action="store_true", default=True)

    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.device("cuda")

    if device_arg == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS is not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_tif_2d(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(f"Only support 2D tif. Got {path} with shape {arr.shape}")

    return arr


def normalize_image(img: np.ndarray, mode: str | None = "zscore") -> np.ndarray:
    img = img.astype(np.float32, copy=False)

    if mode is None or mode == "none":
        return img

    if mode == "zscore":
        mean = float(img.mean())
        std = float(img.std())
        if std < 1e-6:
            return img - mean
        return (img - mean) / std

    if mode == "minmax":
        mn = float(img.min())
        mx = float(img.max())
        if mx - mn < 1e-6:
            return img - mn
        return (img - mn) / (mx - mn)

    raise ValueError(f"Unknown normalize mode: {mode}")


def resolve_image_path(rec: Dict) -> Path:
    for key in ["image_copy_tif", "image_tif"]:
        if key in rec:
            return Path(rec[key])
    raise KeyError("No image tif path found in manifest record.")


def build_model_from_checkpoint_meta(ckpt_meta: Dict, device: torch.device) -> UNetPaper:
    ckpt_args = ckpt_meta.get("args", {}) if isinstance(ckpt_meta, dict) else {}

    in_channels = int(ckpt_args.get("in_channels", 1))
    num_classes = int(ckpt_args.get("num_classes", 2))
    use_bottleneck_dropout = bool(ckpt_args.get("use_bottleneck_dropout", True))
    dropout_p = float(ckpt_args.get("dropout_p", 0.5))

    model = UNetPaper(
        in_channels=in_channels,
        num_classes=num_classes,
        use_bottleneck_dropout=use_bottleneck_dropout,
        bottleneck_dropout_p=dropout_p,
    ).to(device)

    return model


def load_checkpoint_model(ckpt_path: Path, device: torch.device) -> Tuple[UNetPaper, Dict]:
    ckpt = torch.load(str(ckpt_path), map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model = build_model_from_checkpoint_meta(ckpt, device)
        model.load_state_dict(ckpt["model_state"])
        return model, ckpt

    model = UNetPaper(
        in_channels=1,
        num_classes=2,
        use_bottleneck_dropout=True,
        bottleneck_dropout_p=0.5,
    ).to(device)
    model.load_state_dict(ckpt)
    return model, {"model_state_only": True}


def ceil_to_multiple(x: int, base: int) -> int:
    return int(math.ceil(x / base) * base)


@torch.no_grad()
def overlap_tile_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    input_size: int = 572,
    output_size: int = 388,
    normalize: str | None = "zscore",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    overlap-tile inference + mirror padding
    返回:
        pred_binary: H x W, uint8, 0/1
        prob_fg:     H x W, float32, 前景概率
    """
    if input_size <= output_size:
        raise ValueError("input_size must be larger than output_size")

    if (input_size - output_size) % 2 != 0:
        raise ValueError("input_size - output_size must be even")

    margin = (input_size - output_size) // 2

    image = normalize_image(image, normalize)
    H, W = image.shape

    Hp = ceil_to_multiple(H, output_size)
    Wp = ceil_to_multiple(W, output_size)

    pad_bottom = Hp - H
    pad_right = Wp - W

    if pad_bottom > 0 or pad_right > 0:
        image_base = np.pad(
            image,
            ((0, pad_bottom), (0, pad_right)),
            mode="reflect",
        )
    else:
        image_base = image

    image_pad = np.pad(
        image_base,
        ((margin, margin), (margin, margin)),
        mode="reflect",
    )

    prob_fg_full = np.zeros((Hp, Wp), dtype=np.float32)

    model.eval()

    for y in range(0, Hp, output_size):
        for x in range(0, Wp, output_size):
            inp = image_pad[y:y + input_size, x:x + input_size]

            if inp.shape != (input_size, input_size):
                raise ValueError(
                    f"Unexpected input tile shape {inp.shape}, expected {(input_size, input_size)}"
                )

            inp_t = torch.from_numpy(inp[None, None, ...].astype(np.float32)).to(device)

            logits = model(inp_t)
            probs = torch.softmax(logits, dim=1)

            fg = probs[0, 1].detach().cpu().numpy().astype(np.float32)
            prob_fg_full[y:y + output_size, x:x + output_size] = fg

    prob_fg = prob_fg_full[:H, :W]
    pred_binary = (prob_fg >= 0.5).astype(np.uint8)

    return pred_binary, prob_fg


def save_pred_tif(path: Path, pred_binary: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = pred_binary.astype(np.uint8) * 255
    tifffile.imwrite(str(path), vis)


def save_pred_npy(path: Path, pred_binary: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), pred_binary.astype(np.uint8))


def save_prob_npy(path: Path, prob_fg: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), prob_fg.astype(np.float32))


def main() -> None:
    args = parse_args()

    args.processed_root = args.processed_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)

    manifest_dir = args.processed_root / "manifests"
    manifest_path = manifest_dir / args.manifest_name
    processed_summary_path = manifest_dir / "summary.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    records = load_json(manifest_path)
    processed_summary = load_json(processed_summary_path) if processed_summary_path.exists() else {}

    print("========== Test Config ==========")
    print(json.dumps({
        "processed_root": str(args.processed_root),
        "manifest_name": args.manifest_name,
        "checkpoint": str(args.checkpoint),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "input_size": args.input_size,
        "output_size": args.output_size,
        "normalize": args.normalize,
        "num_records": len(records),
    }, indent=2, ensure_ascii=False))
    print("=================================")

    model, ckpt_meta = load_checkpoint_model(args.checkpoint, device)

    pred_tif_dir = args.out_dir / "pred_masks_tif"
    pred_npy_dir = args.out_dir / "pred_masks_npy"
    prob_npy_dir = args.out_dir / "prob_maps_npy"

    meta_json = args.out_dir / "checkpoint_meta.json"
    summary_json = args.out_dir / "summary.json"
    inference_manifest_json = args.out_dir / "inference_manifest.json"

    if isinstance(ckpt_meta, dict):
        meta_json.write_text(
            json.dumps(
                {
                    "checkpoint_path": str(args.checkpoint),
                    "checkpoint_keys": list(ckpt_meta.keys()),
                    "epoch": ckpt_meta.get("epoch", None),
                    "global_step": ckpt_meta.get("global_step", None),
                    "best_loss": ckpt_meta.get("best_loss", None),
                    "args": ckpt_meta.get("args", None),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    inference_records: List[Dict] = []

    t0 = time.time()

    for idx, rec in enumerate(records, start=1):
        image_path = resolve_image_path(rec)
        image = read_tif_2d(image_path).astype(np.float32, copy=False)

        pred, prob = overlap_tile_inference(
            model=model,
            image=image,
            device=device,
            input_size=args.input_size,
            output_size=args.output_size,
            normalize=args.normalize,
        )

        stem = f'{rec["seq"]}_{rec["frame"]}'

        pred_tif_path = pred_tif_dir / f"{stem}.tif"
        pred_npy_path = pred_npy_dir / f"{stem}.npy"
        prob_npy_path = prob_npy_dir / f"{stem}.npy"

        if args.save_pred_tif:
            save_pred_tif(pred_tif_path, pred)
        if args.save_pred_npy:
            save_pred_npy(pred_npy_path, pred)
        if args.save_prob_npy:
            save_prob_npy(prob_npy_path, prob)

        inference_records.append({
            "seq": rec["seq"],
            "frame": rec["frame"],
            "image_path": str(image_path),
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "pred_tif": str(pred_tif_path) if args.save_pred_tif else None,
            "pred_npy": str(pred_npy_path) if args.save_pred_npy else None,
            "prob_npy": str(prob_npy_path) if args.save_prob_npy else None,
        })

        print(f"[{idx:03d}/{len(records):03d}] {stem} inference done.")

    elapsed = time.time() - t0

    inference_manifest_json.write_text(
        json.dumps(inference_records, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "processed_root": str(args.processed_root),
        "manifest_name": args.manifest_name,
        "out_dir": str(args.out_dir),
        "num_images": len(records),
        "elapsed_sec": elapsed,
        "input_size": args.input_size,
        "output_size": args.output_size,
        "normalize": args.normalize,
        "device": str(device),
        "note": (
            "Pure inference only. No segmentation metrics are computed in this script. "
            "Use a separate evaluation/report stage for IoU, Dice, visualization, and analysis."
        ),
        "processed_summary": processed_summary,
    }

    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("========== Inference Summary ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print("=======================================")
    print(f"Saved inference manifest: {inference_manifest_json}")
    print(f"Saved summary JSON      : {summary_json}")


if __name__ == "__main__":
    main()
