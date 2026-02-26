from pathlib import Path
import numpy as np
from PIL import Image

out_dir = Path(r"/Users/dby051225/Desktop/GB Essay 2026/Test/20260212-093015__RESC__RD4AD__baseline_cpu__seed42__debug/06_outputs/results/reverse_distillation/RESC/run/images/vis_b1_80")
idx_file = out_dir / "index__exported_heatmaps.txt"
assert idx_file.is_file(), f"missing {idx_file}"

def load_gray(p: Path):
    a = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    return a

rows = []
for line in idx_file.read_text(encoding="utf-8", errors="ignore").splitlines():
    if not line.strip():
        continue
    cls, img_path, fn_map, fn_ovl = line.split("\t")
    p_map = out_dir / cls / fn_map
    p_ovl = out_dir / cls / fn_ovl
    if not p_map.is_file():
        continue
    m = load_gray(p_map)
    # 一个稳定的图像级分数：top 1% 均值
    flat = m.reshape(-1)
    k = max(1, int(0.01 * flat.size))
    score = float(np.mean(np.partition(flat, -k)[-k:]))
    area = float(np.mean(m > 0.5))  # 0.5 阈值下高响应面积占比
    rows.append((cls, img_path, p_map.name, p_ovl.name, score, area))

norm = [r for r in rows if r[0] == "normal"]
abn  = [r for r in rows if r[0] == "abnormal"]

# 伪响应: normal 里 score 最高的前 10
norm_top = sorted(norm, key=lambda x: x[4], reverse=True)[:10]
# 覆盖弱: abnormal 里 score 最低的前 10（可能是漏检或低对比）
abn_low  = sorted(abn, key=lambda x: x[4])[:10]

# 简单失败定义
# 误报: normal 分数 > abnormal 中位数
abn_med = np.median([r[4] for r in abn]) if abn else 1.0
fp = [r for r in norm if r[4] > abn_med]
# 漏检: abnormal 分数 < normal 95分位
norm_p95 = np.quantile([r[4] for r in norm], 0.95) if norm else 0.0
fn = [r for r in abn if r[4] < norm_p95]

report = []
report.append(f"Total exported: {len(rows)}")
report.append(f"Normal: {len(norm)}  Abnormal: {len(abn)}")
report.append(f"FP count (normal score > abnormal median {abn_med:.4f}): {len(fp)}")
report.append(f"FN count (abnormal score < normal p95 {norm_p95:.4f}): {len(fn)}")
report.append("")
report.append("Top-10 normal pseudo responses (score desc):")
for cls, imgp, mp, ov, s, a in norm_top:
    report.append(f"{s:.4f} area@0.5={a:.4f}  {ov}  src={Path(imgp).name}")
report.append("")
report.append("Bottom-10 abnormal weak coverage (score asc):")
for cls, imgp, mp, ov, s, a in abn_low:
    report.append(f"{s:.4f} area@0.5={a:.4f}  {ov}  src={Path(imgp).name}")

(out_dir / "qual_report.txt").write_text("\n".join(report), encoding="utf-8")
print("written:", out_dir / "qual_report.txt")
