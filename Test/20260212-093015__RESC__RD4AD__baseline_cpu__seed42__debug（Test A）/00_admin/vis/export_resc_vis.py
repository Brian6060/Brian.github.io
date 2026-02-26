import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml
import matplotlib.cm as cm

def to_numpy_map(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.ndim == 4:
            x = x[0]
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x[0]
        return x.numpy()
    return None

def find_map_in(obj):
    if not isinstance(obj, dict):
        return None
    for k in [
        "anomaly_map", "anomaly_maps",
        "pred_mask", "pred_masks",
        "pred_map", "pred_maps",
        "anomaly", "anomaly_score_map",
    ]:
        if k in obj:
            m = to_numpy_map(obj[k])
            if m is not None:
                return m
    return None

def normalize01(m):
    m = m.astype(np.float32)
    m = m - np.min(m)
    mx = np.max(m)
    if mx > 1e-8:
        m = m / mx
    return m

def resize_map01(m01, w, h):
    m_u8 = (m01 * 255).astype(np.uint8)
    im = Image.fromarray(m_u8)
    im = im.resize((w, h), resample=Image.BILINEAR)
    return (np.array(im).astype(np.float32) / 255.0)

def save_one(img_path, amap, out_dir: Path, alpha=0.45):
    img_path = Path(img_path)
    cls = "abnormal" if "ungood" in str(img_path).lower() else "normal"

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img).astype(np.float32)
    H, W = img_np.shape[0], img_np.shape[1]

    amap = np.squeeze(amap)
    m01_raw = normalize01(amap)

    # 保存 raw map（模型分辨率）
    (out_dir / cls).mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    if m01_raw.shape != (H, W):
        raw_u8 = (m01_raw * 255).astype(np.uint8)
        Image.fromarray(raw_u8).save(out_dir / cls / f"anomaly_raw__{stem}.png")

    # resize 到原图分辨率再做 overlay
    m01 = m01_raw if m01_raw.shape == (H, W) else resize_map01(m01_raw, W, H)
    m_u8 = (m01 * 255).astype(np.uint8)

    heat = (cm.get_cmap("jet")(m01)[..., :3] * 255).astype(np.float32)
    overlay = (img_np * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)

    fn_map = f"anomaly__{stem}.png"
    fn_ovl = f"overlay__{stem}.png"

    Image.fromarray(m_u8).save(out_dir / cls / fn_map)
    Image.fromarray(overlay).save(out_dir / cls / fn_ovl)

    return cls, fn_map, fn_ovl

def get_paths_from_dataset(loader, max_items=0):
    ds = getattr(loader, "dataset", None)
    samples = getattr(ds, "samples", None) if ds is not None else None
    if samples is None:
        return None
    try:
        if hasattr(samples, "columns") and "image_path" in samples.columns:
            paths = [str(x) for x in samples["image_path"].tolist()]
            return paths[:max_items] if max_items else paths
    except Exception:
        pass
    return None

def get_paths_by_iterating_loader(loader, max_items=0):
    paths = []
    for batch in loader:
        if isinstance(batch, dict):
            v = batch.get("image_path", None) or batch.get("image_paths", None)
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                paths.extend([str(x) for x in v])
            else:
                paths.append(str(v))
        if max_items and len(paths) >= max_items:
            break
    return paths[:max_items] if max_items else paths

def flatten_predictions(preds, max_items=0):
    maps = []
    for out in preds:
        items = []
        if isinstance(out, dict):
            items = [out]
        elif isinstance(out, (list, tuple)):
            items = list(out)
        for it in items:
            amap = find_map_in(it)
            if amap is not None:
                maps.append(np.squeeze(amap))
            if max_items and len(maps) >= max_items:
                return maps
    return maps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    out_dir = Path(args.out_dir)
    weight = Path(args.weight)
    out_dir.mkdir(parents=True, exist_ok=True)

    from anomalib.config import get_configurable_parameters
    from anomalib.data import get_datamodule
    from anomalib.models import get_model
    from pytorch_lightning import Trainer

    config = get_configurable_parameters(config_path=str(cfg_path))
    datamodule = get_datamodule(config)
    model = get_model(config)

    ckpt = torch.load(str(weight), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    paths = get_paths_from_dataset(loader, args.max_items)
    if paths is None:
        paths = get_paths_by_iterating_loader(loader, args.max_items)

    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    trainer = Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
    preds = trainer.predict(model=model, dataloaders=loader)

    maps = flatten_predictions(preds, args.max_items)

    if len(paths) == 0 or len(maps) == 0:
        print("paths =", len(paths), "maps =", len(maps))
        if len(preds) > 0:
            x = preds[0]
            if isinstance(x, dict):
                print("pred[0] keys:", list(x.keys()))
            elif isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], dict):
                print("pred[0][0] keys:", list(x[0].keys()))
        raise SystemExit(1)

    n = min(len(paths), len(maps), args.max_items or 10**9)
    index_lines = []
    for i in range(n):
        cls, fn_map, fn_ovl = save_one(paths[i], maps[i], out_dir)
        index_lines.append(f"{cls}\t{paths[i]}\t{fn_map}\t{fn_ovl}\n")

    (out_dir / "index__exported_heatmaps.txt").write_text("".join(index_lines), encoding="utf-8")
    print("exported =", n)
    print("out_dir  =", out_dir)

if __name__ == "__main__":
    main()
