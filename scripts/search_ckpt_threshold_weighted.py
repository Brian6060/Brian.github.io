from pathlib import Path
import argparse
import glob
import re
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


def ckpt_iter(path: str) -> int:
    m = re.search(r"_iter_(\d+)\.caffemodel$", path)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--deploy", type=str, required=True)
    parser.add_argument("--ckpt_glob", type=str, required=True)
    parser.add_argument("--thresholds", type=str, default="0.52,0.54,0.56,0.58,0.60")
    parser.add_argument("--min_area", type=int, default=64)
    parser.add_argument("--close_ksize", type=int, default=3)
    parser.add_argument("--fill_holes", action="store_true")
    parser.add_argument("--out_tsv", type=str, required=True)
    args = parser.parse_args()

    split_names = [line.strip() for line in Path(args.split).read_text().splitlines() if line.strip()]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    ckpts = sorted(glob.glob(args.ckpt_glob), key=ckpt_iter)

    results = []

    for ckpt in ckpts:
        net = cv2.dnn.readNetFromCaffe(args.deploy, ckpt)
        for thr in thresholds:
            ious = []
            pes = []

            for name in split_names:
                image_path = Path(args.image_dir) / f"{name}.tif"
                gt_path = Path(args.gt_dir) / f"{name}.tif"

                image_np = load_image(image_path)
                gt_mask = load_mask(gt_path)

                blob = image_np[np.newaxis, np.newaxis, :, :].astype(np.float32)
                net.setInput(blob)
                score = net.forward()

                fg_prob = softmax_channelwise(score[0])[1]
                raw_mask = (fg_prob >= thr).astype(np.uint8)
                pred_mask = postprocess_mask(
                    raw_mask,
                    min_area=args.min_area,
                    close_ksize=args.close_ksize,
                    fill_holes=args.fill_holes,
                )

                ious.append(compute_iou(pred_mask, gt_mask))
                pes.append(compute_pixel_error(pred_mask, gt_mask))

            mean_iou = float(np.mean(ious))
            mean_pe = float(np.mean(pes))
            results.append((ckpt, thr, mean_iou, mean_pe))
            print(f"{Path(ckpt).name} | thr={thr:.2f} | MeanIoU={mean_iou:.6f} | MeanPE={mean_pe:.6f}")

    results.sort(key=lambda x: x[2], reverse=True)

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("checkpoint\tthreshold\tmean_iou\tmean_pixel_error\n")
        for ckpt, thr, miou, mpe in results:
            f.write(f"{ckpt}\t{thr:.2f}\t{miou:.6f}\t{mpe:.6f}\n")

    print("-" * 60)
    print("Best:")
    print(f"checkpoint={results[0][0]}")
    print(f"threshold={results[0][1]:.2f}")
    print(f"mean_iou={results[0][2]:.6f}")
    print(f"mean_pixel_error={results[0][3]:.6f}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
