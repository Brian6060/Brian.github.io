from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import cv2


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def load_image(image_path: Path, target_size=(256, 256)) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.BILINEAR)
    image_np = np.array(image, dtype=np.float32) / 255.0
    return image_np


def load_mask(mask_path: Path, target_size=(256, 256)) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    mask_np = np.array(mask, dtype=np.uint8)
    mask_np = (mask_np > 0).astype(np.uint8)
    return mask_np


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def compute_pixel_error(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred != gt).sum()) / float(gt.size)


def save_gray(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--gt", type=str, default="")
    parser.add_argument("--deploy", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    image_path = Path(args.image)
    gt_path = Path(args.gt) if args.gt else None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_np = load_image(image_path, target_size=(256, 256))

    # OpenCV DNN expects NCHW blob
    blob = image_np[np.newaxis, np.newaxis, :, :].astype(np.float32)

    net = cv2.dnn.readNetFromCaffe(args.deploy, args.weights)
    net.setInput(blob)
    score = net.forward()

    # expected output shape: (1, 1, 256, 256)
    if score.ndim != 4 or score.shape[1] != 1:
        raise RuntimeError(f"Unexpected output shape: {score.shape}")

    score_map = score[0, 0]
    prob_map = sigmoid(score_map)
    pred_mask = (prob_map >= args.threshold).astype(np.uint8)

    stem = image_path.stem
    input_vis = (image_np * 255).astype(np.uint8)
    prob_vis = (prob_map * 255).astype(np.uint8)
    pred_vis = (pred_mask * 255).astype(np.uint8)

    save_gray(input_vis, outdir / f"{stem}_input.png")
    save_gray(prob_vis, outdir / f"{stem}_prob.png")
    save_gray(pred_vis, outdir / f"{stem}_pred.png")

    print(f"Saved input: {outdir / f'{stem}_input.png'}")
    print(f"Saved prob : {outdir / f'{stem}_prob.png'}")
    print(f"Saved pred : {outdir / f'{stem}_pred.png'}")

    if gt_path is not None and gt_path.exists():
        gt_mask = load_mask(gt_path, target_size=(256, 256))
        gt_vis = (gt_mask * 255).astype(np.uint8)
        save_gray(gt_vis, outdir / f"{stem}_gt.png")

        iou = compute_iou(pred_mask, gt_mask)
        pixel_error = compute_pixel_error(pred_mask, gt_mask)

        print(f"IoU: {iou:.6f}")
        print(f"Pixel Error: {pixel_error:.6f}")
        print(f"Saved gt   : {outdir / f'{stem}_gt.png'}")
    else:
        print("No GT provided. Metrics not computed.")


if __name__ == "__main__":
    main()