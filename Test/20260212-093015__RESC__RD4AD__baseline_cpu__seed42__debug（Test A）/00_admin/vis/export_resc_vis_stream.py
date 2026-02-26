import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml
import matplotlib.cm as cm

IMG_EXT = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        x = x.detach().cpu()
        return x.numpy()
    return None

def pick_first(v):
    if isinstance(v, (list, tuple)) and len(v) > 0:
        return v[0]
    return v

def extract_map(out):
    # out 可能是 dict，或 list[dict]，或 tuple
    if isinstance(out, (list, tuple)) and len(out) > 0:
        out = out[0]
    if not isinstance(out, dict):
        return None
    for k in [
        "anomaly_map", "anomaly_maps",
        "pred_mask", "pred_masks",
        "pred_map", "pred_maps",
        "anomaly", "anomaly_score_map",
    ]:
        if k in out:
            m = out[k]
            if torch.is_tensor(m):
                if m.ndim == 4:  # B,1,H,W
                    m = m[0,0]
                elif m.ndim == 3 and m.shape[0] in (1,3):
                    m = m[0]
            m = to_numpy(m)
            if m is not None:
                m = np.squeeze(m)
                return m
    return None

def normalize01(m):
    m = m.astype(np.float32)
    m = m - np.min(m)
    mx = np.max(m)
    if mx > 1e-8:
        m = m / mx
    return m

def resize01(m01, W, H):
    u8 = (m01 * 255).astype(np.uint8)
    im = Image.fromarray(u8).resize((W, H), resample=Image.BILINEAR)
    return np.array(im).astype(np.float32) / 255.0

def save_pair(img_path, amap, out_dir: Path, idx: int, alpha=0.45):
    img_path = Path(img_path)
    cls = "abnormal" if "ungood" in str(img_path).lower() else "normal"

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img).astype(np.float32)
    H, W = img_np.shape[:2]

    m01_raw = normalize01(np.squeeze(amap))
    if m01_raw.shape != (H, W):
        m01 = resize01(m01_raw, W, H)
    else:
        m01 = m01_raw

    m_u8 = (m01 * 255).astype(np.uint8)
    heat = (cm.get_cmap("jet")(m01)[..., :3] * 255).astype(np.float32)
    overlay = (img_np * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)

    (out_dir / cls).mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    prefix = f"{idx:04d}__{stem}"

    Image.fromarray(m_u8).save(out_dir / cls / f"anomaly__{prefix}.png")
    Image.fromarray(overlay).save(out_dir / cls / f"overlay__{prefix}.png")

    return cls, img_path.name, f"anomaly__{prefix}.png", f"overlay__{prefix}.png"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    weight = Path(args.weight)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from anomalib.config import get_configurable_parameters
    from anomalib.data import get_datamodule
    from anomalib.models import get_model

    config = get_configurable_parameters(config_path=str(cfg_path))

    # 强制小 batch，防止 CPU 顶满
    try:
        config.dataset.inference_batch_size = args.batch_size
        config.dataset.eval_batch_size = args.batch_size
        config.dataset.num_workers = 0
    except Exception:
        pass

    datamodule = get_datamodule(config)
    model = get_model(config)

    ckpt = torch.load(str(weight), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    # 先取出 image_path 列表
    ds = loader.dataset
    paths = None
    if hasattr(ds, "samples") and hasattr(ds.samples, "columns") and "image_path" in ds.samples.columns:
        paths = [str(x) for x in ds.samples["image_path"].tolist()]
    if paths is None:
        # 兜底，跑一遍 loader 抽 image_path
        paths = []
        for b in loader:
            if isinstance(b, dict) and ("image_path" in b):
                v = b["image_path"]
                if isinstance(v, (list, tuple)):
                    paths.extend([str(x) for x in v])
                else:
                    paths.append(str(v))
            if len(paths) >= args.start + args.count:
                break

    end = min(len(paths), args.start + args.count)
    sel = paths[args.start:end]

    index = []
    exported = 0

    with torch.no_grad():
        for i, img_path in enumerate(sel):
            # 重新构造一个 batch 给模型 predict_step
            # 最稳做法是从 loader.dataset 读图同一套 transform，但这里直接复用 dataloader batch 更保险
            # 所以我们按索引走 loader，跳过前 start
            pass

    # 按索引遍历 loader，保证输入与模型一致
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    cur = 0
    tgt_lo = args.start
    tgt_hi = args.start + len(sel)

    with torch.no_grad():
        for batch in loader:
            if cur >= tgt_hi:
                break
            if cur < tgt_lo:
                cur += 1
                continue

            # batch_size=1 预期
            img_path = None
            if isinstance(batch, dict):
                img_path = batch.get("image_path", None)
            img_path = pick_first(img_path)
            if img_path is None:
                cur += 1
                continue

            if hasattr(model, "predict_step"):
                out = model.predict_step(batch, cur)
            else:
                # 兜底
                x = batch.get("image", None) if isinstance(batch, dict) else None
                out = model(x)

            amap = extract_map(out)
            if amap is None:
                # 打印一次 key 便于适配
                if isinstance(out, dict):
                    print("predict out keys:", list(out.keys()))
                elif isinstance(out, (list, tuple)) and len(out) and isinstance(out[0], dict):
                    print("predict out[0] keys:", list(out[0].keys()))
                raise SystemExit("cannot find anomaly map in predict output")

            cls, src_name, fn_map, fn_ovl = save_pair(img_path, amap, out_dir, cur)
            index.append(f"{cls}\t{img_path}\t{fn_map}\t{fn_ovl}\n")
            exported += 1

            if exported % 5 == 0:
                print("exported", exported, "files now:", sum(1 for _ in out_dir.rglob("overlay__*.png")))

            cur += 1

    (out_dir / "index__exported_heatmaps.txt").write_text("".join(index), encoding="utf-8")
    print("DONE exported =", exported)
    print("out_dir =", out_dir)

if __name__ == "__main__":
    main()
