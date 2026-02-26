#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-$PWD}"
cd "$RUN_DIR"
RUN_TAG="$(basename "$RUN_DIR")"
TS="$(date +%Y%m%d-%H%M%S)"

# 目录骨架
mkdir -p \
  00_admin/index 00_admin/scripts \
  01_env \
  02_src \
  03_data \
  04_configs/"$RUN_TAG" \
  05_logs/"$RUN_TAG" \
  06_outputs/"$RUN_TAG"/checkpoints \
  06_outputs/"$RUN_TAG"/metrics \
  06_outputs/"$RUN_TAG"/results_engine \
  06_outputs/"$RUN_TAG"/images \
  07_figures/"$RUN_TAG"/qual \
  07_figures/"$RUN_TAG"/paper_ready/shortlist \
  07_figures/"$RUN_TAG"/paper_ready/selected \
  08_docs/paper_assets/"$RUN_TAG"/{quant,qual,repro,configs,logs,ckpt,notes} \
  99_tmp

# 清理垃圾
find . -name ".DS_Store" -type f -delete || true
find . -name "__MACOSX" -type d -prune -exec rm -rf {} + || true
find . -name "._*" -type f -delete || true

# 根目录散落文件粗分类归位（保守规则，只处理明显类型）
shopt -s nullglob

for f in ./*.ckpt; do mv "$f" "06_outputs/$RUN_TAG/checkpoints/" || true; done
for f in ./*metrics*.csv ./*metrics*.json; do mv "$f" "06_outputs/$RUN_TAG/metrics/" || true; done
for f in ./*.log; do mv "$f" "05_logs/$RUN_TAG/" || true; done
for f in ./*.yml ./*.yaml ./*.json; do mv "$f" "04_configs/$RUN_TAG/" || true; done
for f in ./*.png ./*.jpg ./*.jpeg; do mv "$f" "07_figures/$RUN_TAG/qual/" || true; done

# 兼容：如果存在 figures 目录，统一归到 07_figures
if [ -d "figures" ]; then
  mv "figures" "07_figures/$RUN_TAG/qual/figures_raw__${TS}" || true
fi

# 修复你之前出现的 nested run：只做“拉平 + 指向 canon”
CANON="06_outputs/$RUN_TAG/results_engine/reverse_distillation__RESC__run"
NESTED="06_outputs/results/reverse_distillation/RESC/$RUN_TAG/reverse_distillation/RESC/run"
FLAT="06_outputs/results/reverse_distillation/RESC/$RUN_TAG/run"

if [ -d "$NESTED" ]; then
  mkdir -p "00_admin"
  if [ -d "$CANON" ]; then
    diff -qr "$CANON" "$NESTED" > "00_admin/diff__canon_vs_nested__${TS}.txt" || true
    if diff -qr "$CANON" "$NESTED" >/dev/null 2>&1; then
      rm -rf "$NESTED"
    else
      mkdir -p "99_tmp"
      mv "$NESTED" "99_tmp/nested_run__${TS}"
      mkdir -p "$(dirname "$NESTED")"
    fi
    python3 - <<PY
import os
canon="${CANON}"
nested="${NESTED}"
flat="${FLAT}"
for p in [nested, flat]:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.lexists(p):
        try: os.remove(p)
        except: pass
    rel=os.path.relpath(canon, os.path.dirname(p))
    os.symlink(rel, p)
print("linked nested+flat -> canon")
PY
  fi
fi

# paper_assets 用软链接聚合（不复制大文件）
ASSET="08_docs/paper_assets/$RUN_TAG"
ln -sfn "../../../06_outputs/$RUN_TAG/metrics" "$ASSET/quant/metrics_dir" 2>/dev/null || true
ln -sfn "../../../06_outputs/$RUN_TAG/checkpoints/model.ckpt" "$ASSET/ckpt/model.ckpt" 2>/dev/null || true
ln -sfn "../../../04_configs/$RUN_TAG" "$ASSET/configs/configs_dir" 2>/dev/null || true
ln -sfn "../../../05_logs/$RUN_TAG" "$ASSET/logs/logs_dir" 2>/dev/null || true
ln -sfn "../../../07_figures/$RUN_TAG" "$ASSET/qual/figures_dir" 2>/dev/null || true
ln -sfn "../../../06_outputs/$RUN_TAG/results_engine" "$ASSET/qual/results_engine_dir" 2>/dev/null || true

# 可视化候选池与 shortlist
python3 - <<'PY'
import os, re, csv, glob
run_dir=os.getcwd()
run_tag=os.path.basename(run_dir)

# 递归找 png
roots=[
  os.path.join(run_dir,"06_outputs",run_tag,"results_engine"),
  os.path.join(run_dir,"07_figures",run_tag),
]
pngs=[]
for r in roots:
  if os.path.isdir(r):
    pngs += glob.glob(os.path.join(r,"**","*.png"), recursive=True)

def score_from_name(p):
  fn=os.path.basename(p)
  m=re.search(r"__([0-9]+)_([0-9]+)\.png$", fn)
  if not m:
    return None
  return float(f"{m.group(1)}.{m.group(2)}")

def group_from_path(p):
  lp=p.lower()
  if "/abnormal/" in lp or "\\abnormal\\" in lp: return "abnormal"
  if "/normal/" in lp or "\\normal\\" in lp: return "normal"
  if "heat" in lp or "anomaly" in lp or "map" in lp: return "heatmap_misc"
  return "misc"

rows=[]
for p in pngs:
  rows.append((os.path.relpath(p, run_dir), group_from_path(p), score_from_name(p)))

idx=os.path.join(run_dir,"00_admin","index",f"vis_candidates__{run_tag}.csv")
os.makedirs(os.path.dirname(idx), exist_ok=True)
with open(idx,"w",newline="") as f:
  w=csv.writer(f); w.writerow(["path","group","score"])
  for r in rows: w.writerow([r[0], r[1], "" if r[2] is None else r[2]])

short_base=os.path.join(run_dir,"07_figures",run_tag,"paper_ready","shortlist")
os.makedirs(short_base, exist_ok=True)

def symlink(src, dst):
  os.makedirs(os.path.dirname(dst), exist_ok=True)
  if os.path.lexists(dst): os.remove(dst)
  rel=os.path.relpath(os.path.join(run_dir,src), os.path.dirname(dst))
  os.symlink(rel, dst)

def topk(group, k, reverse=True):
  xs=[r for r in rows if r[1]==group and r[2] is not None]
  xs.sort(key=lambda x:x[2], reverse=reverse)
  return xs[:k]

# 三组：异常高分，正常高分伪阳性，正常低分
for name, grp, k, rev in [
  ("abnormal_top12","abnormal",12,True),
  ("normal_fp_top12","normal",12,True),
  ("normal_low12","normal",12,False),
]:
  out=os.path.join(short_base,name)
  os.makedirs(out, exist_ok=True)
  for i,(p,g,s) in enumerate(topk(grp,k,rev)):
    bn=os.path.basename(p)
    symlink(p, os.path.join(out, f"{i:02d}__{bn}"))

print("candidate_index:", idx)
print("shortlist_dir:", short_base)
PY

# 生成说明文件
cat > "08_docs/paper_assets/$RUN_TAG/notes/where_to_find.md" <<EOF
RUN_TAG: $RUN_TAG
generated_at: $TS

Quant:
- metrics dir: 06_outputs/$RUN_TAG/metrics/
- main metrics file (current run): metrics__RESC__RD4AD__seed42__debug.csv

Repro:
- configs: 04_configs/$RUN_TAG/
- logs: 05_logs/$RUN_TAG/
- ckpt:  06_outputs/$RUN_TAG/checkpoints/model.ckpt

Qual:
- shortlist: 07_figures/$RUN_TAG/paper_ready/shortlist/
- selected:  07_figures/$RUN_TAG/paper_ready/selected/
- full pool: 06_outputs/$RUN_TAG/results_engine/ (recursive png)
EOF

echo "OK"
echo "shortlist: 07_figures/$RUN_TAG/paper_ready/shortlist"
echo "paper_assets: 08_docs/paper_assets/$RUN_TAG"
