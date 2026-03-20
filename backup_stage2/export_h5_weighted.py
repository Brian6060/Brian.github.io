from pathlib import Path
import numpy as np
import h5py
from PIL import Image
from scipy.ndimage import distance_transform_edt

REAL_ROOT = Path("/Users/dby051225/Desktop/VIS/U-Net/U-Net Experiment Reproduction")
ALIAS_ROOT = Path("/Users/dby051225/Desktop/VIS/U-Net/U-Net_Experiment_Reproduction")

PROC_IMG_DIR = REAL_ROOT / "data" / "processed" / "images"
RAW_INST_DIR = REAL_ROOT / "data" / "raw" / "segmentation_maps"
SPLIT_DIR = REAL_ROOT / "data" / "splits"

H5_DIR_REAL = REAL_ROOT / "data" / "h5_weighted"
H5_DIR_ALIAS = ALIAS_ROOT / "data" / "h5_weighted"

H5_DIR_REAL.mkdir(parents=True, exist_ok=True)
H5_DIR_ALIAS.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (256, 256)
W0 = 10.0
SIGMA = 5.0


def load_split(split_name: str):
    split_file = SPLIT_DIR / f"{split_name}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    if len(names) == 0:
        raise ValueError(f"No sample names found in {split_file}")
    return names


def load_processed_image(sample_name: str) -> np.ndarray:
    img_path = PROC_IMG_DIR / f"{sample_name}.tif"
    if not img_path.exists():
        raise FileNotFoundError(f"Processed image not found: {img_path}")

    image = Image.open(img_path).convert("L")
    if image.size != TARGET_SIZE:
        image = image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)

    image_np = np.array(image, dtype=np.float32) / 255.0
    return image_np


def load_resized_instance_mask(sample_name: str) -> np.ndarray:
    inst_path = RAW_INST_DIR / f"{sample_name}.tif"
    if not inst_path.exists():
        raise FileNotFoundError(f"Raw instance mask not found: {inst_path}")

    inst = np.array(Image.open(inst_path))
    if inst.ndim != 2:
        raise ValueError(f"Instance mask should be 2D, got shape {inst.shape} for {inst_path}")

    # 保留实例 id，不能 convert('L')
    inst_img = Image.fromarray(inst.astype(np.int32))
    if inst_img.size != TARGET_SIZE:
        inst_img = inst_img.resize(TARGET_SIZE, Image.Resampling.NEAREST)

    inst_resized = np.array(inst_img, dtype=np.int32)
    return inst_resized


def compute_class_weights(instance_dict: dict):
    fg_pixels = 0
    total_pixels = 0

    for inst in instance_dict.values():
        fg_pixels += int((inst > 0).sum())
        total_pixels += int(inst.size)

    fg_ratio = fg_pixels / max(total_pixels, 1)
    bg_ratio = 1.0 - fg_ratio

    w_fg = 1.0 / max(fg_ratio, 1e-6)
    w_bg = 1.0 / max(bg_ratio, 1e-6)

    return w_bg, w_fg, fg_ratio, bg_ratio


def build_label_and_weight_map(instance_mask: np.ndarray, w_bg: float, w_fg: float,
                               w0: float = 10.0, sigma: float = 5.0):
    binary_label = (instance_mask > 0).astype(np.uint8)
    h, w = binary_label.shape

    wc = np.zeros((h, w), dtype=np.float32)
    wc[binary_label == 0] = w_bg
    wc[binary_label == 1] = w_fg

    ids = np.unique(instance_mask)
    ids = ids[ids != 0]

    if len(ids) >= 2:
        dist_list = []
        for obj_id in ids:
            obj = (instance_mask == obj_id)
            dist = distance_transform_edt(~obj).astype(np.float32)
            dist_list.append(dist)

        dists = np.stack(dist_list, axis=0)   # K x H x W
        dists.sort(axis=0)

        d1 = dists[0]
        d2 = dists[1]

        border_term = w0 * np.exp(-((d1 + d2) ** 2) / (2.0 * sigma ** 2))
        border_term = border_term * (binary_label == 0)
    else:
        border_term = np.zeros((h, w), dtype=np.float32)

    weight_map = wc + border_term.astype(np.float32)

    label = binary_label.astype(np.float32)
    weight_map = weight_map.astype(np.float32)

    return label, weight_map


def export_split(split_name: str):
    names = load_split(split_name)

    instance_dict = {}
    for name in names:
        instance_dict[name] = load_resized_instance_mask(name)

    w_bg, w_fg, fg_ratio, bg_ratio = compute_class_weights(instance_dict)

    image_list = []
    label_list = []
    weight_list = []

    for name in names:
        image_np = load_processed_image(name)
        label_np, weight_np = build_label_and_weight_map(
            instance_dict[name],
            w_bg=w_bg,
            w_fg=w_fg,
            w0=W0,
            sigma=SIGMA,
        )

        image_list.append(image_np[np.newaxis, :, :])     # 1 x H x W
        label_list.append(label_np[np.newaxis, :, :])     # 1 x H x W
        weight_list.append(weight_np[np.newaxis, :, :])   # 1 x H x W

    data_arr = np.stack(image_list, axis=0).astype(np.float32)
    label_arr = np.stack(label_list, axis=0).astype(np.float32)
    weight_arr = np.stack(weight_list, axis=0).astype(np.float32)

    h5_real_path = H5_DIR_REAL / f"{split_name}.h5"
    with h5py.File(h5_real_path, "w") as f:
        f.create_dataset("data", data=data_arr, dtype="float32")
        f.create_dataset("label", data=label_arr, dtype="float32")
        f.create_dataset("weight", data=weight_arr, dtype="float32")

    # list 文件必须写 alias 路径，避免空格
    h5_alias_path = H5_DIR_ALIAS / f"{split_name}.h5"
    list_file = H5_DIR_REAL / f"{split_name}_h5_list.txt"
    with open(list_file, "w") as f:
        f.write(str(h5_alias_path) + "\n")

    print(f"[{split_name}] saved: {h5_real_path}")
    print(f"[{split_name}] data shape   : {data_arr.shape}, dtype: {data_arr.dtype}")
    print(f"[{split_name}] label shape  : {label_arr.shape}, dtype: {label_arr.dtype}")
    print(f"[{split_name}] weight shape : {weight_arr.shape}, dtype: {weight_arr.dtype}")
    print(f"[{split_name}] class ratios : fg={fg_ratio:.6f}, bg={bg_ratio:.6f}")
    print(f"[{split_name}] weight stats : min={float(weight_arr.min()):.6f}, max={float(weight_arr.max()):.6f}, mean={float(weight_arr.mean()):.6f}")
    print(f"[{split_name}] list file    : {list_file}")
    print(f"[{split_name}] alias h5 path: {h5_alias_path}")


if __name__ == "__main__":
    export_split("train")
    export_split("val")
