import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def list_images(d: str):
    files = []
    for p in sorted(Path(d).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            files.append(str(p))
    return files

def preprocess(img_path: str, image_size: int, normalization: str):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    x = np.asarray(img).astype(np.float32) / 255.0  # HWC
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = torch.from_numpy(x)

    if normalization and normalization.lower() == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std

    return x, img

def to_uint8_heatmap(am: np.ndarray):
    a = am.astype(np.float32)
    a = a - np.nanmin(a)
    den = np.nanmax(a) + 1e-8
    a = (a / den * 255.0).clip(0, 255).astype(np.uint8)
    return a

def overlay_simple(rgb_pil: Image.Image, heat_u8: np.ndarray):
    base = np.asarray(rgb_pil).astype(np.uint8)
    ov = base.copy()
    ov[..., 0] = np.maximum(ov[..., 0], heat_u8)
    return ov

def extract_anomaly_map(out):
    if isinstance(out, dict):
        for k in ("anomaly_map","anomaly_maps","anomaly_score_map","pred_mask","pred_masks"):
            if k in out:
                return out[k], list(out.keys())
        for k,v in out.items():
            if torch.is_tensor(v):
                return v, list(out.keys())
        return None, list(out.keys())

    if torch.is_tensor(out):
        return out, None

    if isinstance(out, (list, tuple)):
        for v in out:
            if torch.is_tensor(v):
                return v, None
    return None, None

def build_model(cfg):
    from anomalib.models import get_model
    if not OmegaConf.select(cfg, "model.input_size"):
        cfg.model.input_size = (int(cfg.dataset.image_size), int(cfg.dataset.image_size))
    return get_model(config=cfg)

def load_ckpt(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=False)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_normal", type=int, default=5)
    ap.add_argument("--n_abnormal", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)

    normal_dir = str(cfg.dataset.normal_test_dir)
    abnormal_dir = str(cfg.dataset.abnormal_dir)
    image_size = int(cfg.dataset.image_size)
    normalization = str(cfg.dataset.normalization)

    normals = list_images(normal_dir)
    abnormals = list_images(abnormal_dir)

    if len(normals) == 0:
        raise SystemExit(f"normal_test_dir empty: {normal_dir}")
    if len(abnormals) == 0:
        raise SystemExit(f"abnormal_dir empty: {abnormal_dir}")

    normals = normals[: args.n_normal]
    abnormals = abnormals[: args.n_abnormal]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg)
    load_ckpt(model, args.weight)
    model.eval()

    print("ckpt:", args.weight)
    print("normal_dir:", normal_dir)
    print("abnormal_dir:", abnormal_dir)

    index_lines = []
    with torch.no_grad():
        for tag, files in [("normal", normals), ("abnormal", abnormals)]:
            for idx, fp in enumerate(files):
                x, rgb = preprocess(fp, image_size=image_size, normalization=normalization)
                x = x.unsqueeze(0)  # 1x3xHxW

                try:
                    out = model(x)
                except Exception:
                    out = model({'image': x})
                am, keys = extract_anomaly_map(out)
                if am is None:
                    print("output keys:", keys)
                    raise SystemExit("cannot find anomaly_map key. see keys above.")

                if torch.is_tensor(am):
                    am = am.detach().cpu().numpy()

                am = np.asarray(am)
                if am.ndim == 4:
                    am = am[0, 0]
                elif am.ndim == 3:
                    am = am[0]
                elif am.ndim != 2:
                    raise SystemExit(f"unexpected anomaly_map shape: {am.shape}")

                heat = to_uint8_heatmap(am)
                ov = overlay_simple(rgb, heat)

                stem = f"{tag}_{idx:02d}"
                heat_path = outdir / f"anomaly_{stem}.png"
                ov_path = outdir / f"overlay_{stem}.png"

                Image.fromarray(heat).save(heat_path)
                Image.fromarray(ov).save(ov_path)

                index_lines.append(f"{tag}\t{fp}\t{heat_path.name}\t{ov_path.name}\n")
                print("saved:", heat_path.name, ov_path.name)

    (outdir / "index__exported_heatmaps.txt").write_text("".join(index_lines))
    print("index:", outdir / "index__exported_heatmaps.txt")

if __name__ == "__main__":
    main()
