from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import binary_fill_holes


def softmax_channelwise(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def load_image(image_path: Path, target_size=(256, 256)) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.BILINEAR)
    return np.array(image, dtype=np.float32) / 255.0


def load_mask(mask_path: Path, target_size=(256, 256)) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    return (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out


def postprocess_mask(mask: np.ndarray, min_area: int = 64, close_ksize: int = 3, fill_holes: bool = True) -> np.ndarray:
    out = mask.astype(np.uint8)

    if close_ksize > 0:
        kernel = np.ones((close_ksize, close_ksize), dtype=np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    out = remove_small_components(out, min_area=min_area)

    if fill_holes:
        out = binary_fill_holes(out > 0).astype(np.uint8)

    return out.astype(np.uint8)


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
    parser.add_argument("--threshold", type=float, default=0.56)
    parser.add_argument("--min_area", type=int, default=64)
    parser.add_argument("--close_ksize", type=int, default=3)
    parser.add_argument("--fill_holes", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image)
    gt_path = Path(args.gt) if args.gt else None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_np = load_image(image_path, target_size=(256, 256))
    blob = image_np[np.newaxis, np.newaxis, :, :].astype(np.float32)

    net = cv2.dnn.readNetFromCaffe(args.deploy, args.weights)
    net.setInput(blob)
    score = net.forward()

    if score.ndim != 4 or score.shape[1] != 2:
        raise RuntimeError(f"Unexpected output shape: {score.shape}")

    score_map = score[0]
    prob_map = softmax_channelwise(score_map)
    fg_prob = prob_map[1]

    raw_mask = (fg_prob >= args.threshold).astype(np.uint8)
    pred_mask = postprocess_mask(
        raw_mask,
        min_area=args.min_area,
        close_ksize=args.close_ksize,
        fill_holes=args.fill_holes,
    )

    stem = image_path.stem

    save_gray((image_np * 255).astype(np.uint8), outdir / f"{stem}_input.png")
    save_gray((fg_prob * 255).astype(np.uint8), outdir / f"{stem}_prob_fg.png")
    save_gray((raw_mask * 255).astype(np.uint8), outdir / f"{stem}_pred_raw.png")
    save_gray((pred_mask * 255).astype(np.uint8), outdir / f"{stem}_pred_post.png")

    print(f"Saved input     : {outdir / f'{stem}_input.png'}")
    print(f"Saved prob      : {outdir / f'{stem}_prob_fg.png'}")
    print(f"Saved raw pred  : {outdir / f'{stem}_pred_raw.png'}")
    print(f"Saved post pred : {outdir / f'{stem}_pred_post.png'}")

    print("score min/max/mean:", float(score_map.min()), float(score_map.max()), float(score_map.mean()))
    print("fg prob min/max/mean:", float(fg_prob.min()), float(fg_prob.max()), float(fg_prob.mean()))
    print("raw foreground pixels:", int(raw_mask.sum()))
    print("raw foreground ratio:", float(raw_mask.mean()))
    print("post foreground pixels:", int(pred_mask.sum()))
    print("post foreground ratio:", float(pred_mask.mean()))

    if gt_path is not None and gt_path.exists():
        gt_mask = load_mask(gt_path, target_size=(256, 256))
        save_gray((gt_mask * 255).astype(np.uint8), outdir / f"{stem}_gt.png")

        iou = compute_iou(pred_mask, gt_mask)
        pixel_error = compute_pixel_error(pred_mask, gt_mask)

        print("gt foreground pixels:", int(gt_mask.sum()))
        print("gt foreground ratio:", float(gt_mask.mean()))
        print(f"IoU: {iou:.6f}")
        print(f"Pixel Error: {pixel_error:.6f}")
        print(f"Saved gt       : {outdir / f'{stem}_gt.png'}")
    else:
        print("No GT provided. Metrics not computed.")


if __name__ == "__main__":
    main()
