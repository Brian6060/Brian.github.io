from pathlib import Path
import numpy as np
from PIL import Image

out_dir = Path(r"/Users/dby051225/Desktop/GB Essay 2026/Test/20260212-093015__RESC__RD4AD__baseline_cpu__seed42__debug/06_outputs/results/reverse_distillation/RESC/run/images/vis_b1_80")
assert out_dir.is_dir(), out_dir

def load_gray(p: Path):
    a = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    return a

def score_map(m):
    flat = m.reshape(-1)
    k = max(1, int(0.01 * flat.size))  # top 1% mean
    s = float(np.mean(np.partition(flat, -k)[-k:]))
    area = float(np.mean(m > 0.5))     # area ratio at threshold 0.5
    return s, area

def src_from_name(name: str):
    # anomaly__0007__8_47.png -> 8_47.png
    parts = name.split("__", 2)
    if len(parts) == 3:
        return parts[2]
    return name.replace("anomaly__", "").replace("overlay__", "")

rows = []
for cls in ["normal", "abnormal"]:
    for p in (out_dir/cls).glob("anomaly__*.png"):
        m = load_gray(p)
        s, a = score_map(m)
        ov = p.name.replace("anomaly__", "overlay__")
        ovp = out_dir/cls/ov
        rows.append((cls, p.name, ov if ovp.is_file() else "", s, a, src_from_name(p.name)))

norm = [r for r in rows if r[0] == "normal"]
abn  = [r for r in rows if r[0] == "abnormal"]

norm_top = sorted(norm, key=lambda x: x[3], reverse=True)[:10]
abn_low  = sorted(abn, key=lambda x: x[3])[:10]

abn_med = float(np.median([r[3] for r in abn])) if abn else 1.0
norm_p95 = float(np.quantile([r[3] for r in norm], 0.95)) if norm else 0.0

fp = [r for r in norm if r[3] > abn_med]
fn = [r for r in abn if r[3] < norm_p95]

report = []
report.append(f"Total overlays: normal={len(norm)} abnormal={len(abn)} total={len(norm)+len(abn)}")
report.append(f"FP count (normal score > abnormal median {abn_med:.4f}): {len(fp)}")
report.append(f"FN count (abnormal score < normal p95 {norm_p95:.4f}): {len(fn)}")
report.append("")
report.append("Top-10 normal pseudo responses (score desc):")
for cls, amap, ov, s, a, src in norm_top:
    report.append(f"{s:.4f} area@0.5={a:.4f}  {ov}  src={src}")
report.append("")
report.append("Bottom-10 abnormal weak coverage (score asc):")
for cls, amap, ov, s, a, src in abn_low:
    report.append(f"{s:.4f} area@0.5={a:.4f}  {ov}  src={src}")

(out_dir / "qual_report_full.txt").write_text("\n".join(report), encoding="utf-8")
print("written:", out_dir / "qual_report_full.txt")
