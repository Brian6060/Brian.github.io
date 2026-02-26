#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-$PWD}"
cd "$RUN_DIR"
RUN_TAG="$(basename "$PWD")"
TS="$(date +%Y%m%d-%H%M%S)"

ARCH="${ARCH:-$RUN_DIR/99_tmp/pruned__20260225-113429}"

if [ ! -d "$ARCH" ]; then
  echo "ARCH not found: $ARCH"
  exit 1
fi

CONFLICT="$RUN_DIR/99_tmp/restore_conflicts__${TS}"
mkdir -p "$CONFLICT"

move_with_backup() {
  local src="$1"
  local dst="$2"
  if [ ! -e "$src" ]; then
    echo "skip missing: $src"
    return 0
  fi
  if [ -e "$dst" ]; then
    mkdir -p "$CONFLICT"
    local bn
    bn="$(basename "$dst")"
    echo "backup existing -> $CONFLICT/$bn"
    mv "$dst" "$CONFLICT/$bn"
  fi
  mkdir -p "$(dirname "$dst")"
  echo "move $src -> $dst"
  mv "$src" "$dst"
}

echo "RUN_TAG=$RUN_TAG"
echo "ARCH=$ARCH"
echo "CONFLICT=$CONFLICT"

# 1) 恢复 raw results
# 归档里 results 对应你之前被清理的 06_outputs/results
move_with_backup "$ARCH/results" "$RUN_DIR/06_outputs/results"

# 2) 恢复 qual 图池
# 归档里 qual 对应你之前被清理的 07_figures/<RUN_TAG>/qual
move_with_backup "$ARCH/qual" "$RUN_DIR/07_figures/${RUN_TAG}/qual"

# 3) 恢复 paper_assets
# 归档里 paper_assets 对应 08_docs/paper_assets
move_with_backup "$ARCH/paper_assets" "$RUN_DIR/08_docs/paper_assets"

# 4) 重建 canonical results_engine 指针
CANON_RUN="$RUN_DIR/06_outputs/${RUN_TAG}/results_engine/reverse_distillation__RESC__run"
RAW_RUN="$RUN_DIR/06_outputs/results/reverse_distillation/RESC/${RUN_TAG}"

mkdir -p "$(dirname "$CANON_RUN")"

if [ -d "$RAW_RUN" ]; then
  rm -f "$CANON_RUN" 2>/dev/null || true
  ln -s "$RAW_RUN" "$CANON_RUN"
else
  echo "WARN: RAW_RUN not found: $RAW_RUN"
fi

# 5) 重建 03_data 内 BMAD run 的软链接，指向 canonical run
DATA_LINK="$RUN_DIR/03_data/BMAD/datasets/RESC/reverse_distillation/RESC/run"
if [ -d "$RAW_RUN" ]; then
  mkdir -p "$(dirname "$DATA_LINK")"
  rm -rf "$DATA_LINK" 2>/dev/null || true
  # 相对链接更稳
  python3 - <<PY
import os
run_dir=os.getcwd()
run_tag=os.path.basename(run_dir)
data_link=os.path.join(run_dir,"03_data","BMAD","datasets","RESC","reverse_distillation","RESC","run")
canon_run=os.path.join(run_dir,"06_outputs",run_tag,"results_engine","reverse_distillation__RESC__run")
rel=os.path.relpath(canon_run, os.path.dirname(data_link))
os.symlink(rel, data_link)
print("linked:", data_link, "->", rel)
PY
fi

# 6) 让 A 实验候选入口 images_dir 指向 canonical images
A_DIR="$RUN_DIR/09_paper_A__sanity__${RUN_TAG}"
CAND="$A_DIR/03_visualization/candidates"
mkdir -p "$CAND"
rm -f "$CAND/images_dir" 2>/dev/null || true

CANON_IMG="$RUN_DIR/06_outputs/${RUN_TAG}/results_engine/reverse_distillation__RESC__run/images"
if [ -d "$CANON_IMG" ]; then
  python3 - <<PY
import os
run_dir=os.getcwd()
run_tag=os.path.basename(run_dir)
a_dir=os.path.join(run_dir,f"09_paper_A__sanity__{run_tag}")
cand=os.path.join(a_dir,"03_visualization","candidates")
img=os.path.join(run_dir,"06_outputs",run_tag,"results_engine","reverse_distillation__RESC__run","images")
dst=os.path.join(cand,"images_dir")
if os.path.lexists(dst): os.remove(dst)
rel=os.path.relpath(img, os.path.dirname(dst))
os.symlink(rel, dst)
print("linked:", dst, "->", rel)
PY
else
  echo "WARN: CANON_IMG not found: $CANON_IMG"
fi

echo "VERIFY"
echo "png in raw results images:"
find "$RUN_DIR/06_outputs/results" -type f -iname "*.png" | wc -l || true
echo "png in canonical images:"
find "$CANON_IMG" -type f -iname "*.png" | wc -l || true

echo "DONE"
echo "Open candidates:"
echo "open \"$CAND/images_dir\""
