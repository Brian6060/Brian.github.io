from pathlib import Path
import numpy as np
import h5py
from PIL import Image

ROOT_DIR = Path("/Users/dby051225/Desktop/VIS/U-Net/U-Net Experiment Reproduction")
IMG_DIR = ROOT_DIR / "data" / "processed" / "images"
SEG_DIR = ROOT_DIR / "data" / "processed" / "segmentation_maps"
SPLIT_DIR = ROOT_DIR / "data" / "splits"
H5_DIR = ROOT_DIR / "data" / "h5"

H5_DIR.mkdir(parents=True, exist_ok=True)

def to_project_relative(path: Path) -> Path:
    return path.relative_to(ROOT_DIR)

def load_split(split_name):
    split_file = SPLIT_DIR / f"{split_name}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]

    if len(names) == 0:
        raise ValueError(f"No sample names found in {split_file}")

    return names

def load_sample(sample_name):
    img_path = IMG_DIR / f"{sample_name}.tif"
    seg_path = SEG_DIR / f"{sample_name}.tif"

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation map not found: {seg_path}")

    image = Image.open(img_path).convert("L")
    seg = Image.open(seg_path).convert("L")

    image = np.array(image, dtype=np.float32) / 255.0
    seg = np.array(seg, dtype=np.uint8)
    seg = (seg > 0).astype(np.float32)

    image = np.expand_dims(image, axis=0)
    seg = np.expand_dims(seg, axis=0)

    return image, seg

def export_h5(split_name):
    names = load_split(split_name)

    images = []
    labels = []

    for name in names:
        image, seg = load_sample(name)
        images.append(image)
        labels.append(seg)

    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0)

    h5_path = H5_DIR / f"{split_name}.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data", data=images)
        f.create_dataset("label", data=labels)

    list_path = H5_DIR / f"{split_name}_h5_list.txt"
    with open(list_path, "w") as f:
        # Caffe's HDF5Data layer treats each line as a raw path token.
        # Writing a project-relative path avoids breakage from spaces in
        # the absolute workspace path.
        f.write(str(to_project_relative(h5_path)) + "\n")

    print(f"[{split_name}] saved: {h5_path}")
    print(f"[{split_name}] data shape: {images.shape}, dtype: {images.dtype}")
    print(f"[{split_name}] label shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"[{split_name}] list file: {list_path}")

if __name__ == "__main__":
    export_h5("train")
    export_h5("val")
