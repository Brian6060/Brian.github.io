#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-$PWD}"
cd "$RUN_DIR"
RUN_TAG="$(basename "$PWD")"
TS="$(date +%Y%m%d-%H%M%S)"

# 默认归档而不是删除
PRUNE_DELETE="${PRUNE_DELETE:-0}"   # 0: move to 99_tmp, 1: rm -rf
DRY_RUN="${DRY_RUN:-0}"            # 1:只打印不执行

A_DIR="09_paper_A__sanity__${RUN_TAG}"
ARCH="99_tmp/pruned__${TS}"
mkdir -p "$ARCH"

run() {
  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY] $*"
  else
    eval "$@"
  fi
}

move_or_delete() {
  local p="$1"
  if [ ! -e "$p" ] && [ ! -L "$p" ]; then
    return 0
  fi
  if [ "$PRUNE_DELETE" = "1" ]; then
    run "rm -rf \"$p\""
  else
    run "mv \"$p\" \"$ARCH/\""
  fi
}

echo "RUN_TAG=$RUN_TAG"
echo "A_DIR=$A_DIR"
echo "MODE: PRUNE_DELETE=$PRUNE_DELETE DRY_RUN=$DRY_RUN"
echo

# 0) A 产物必须存在
if [ ! -d "$A_DIR" ]; then
  echo "Missing $A_DIR. Stop."
  exit 1
fi

# 1) 删除纯垃圾文件（直接删）
run "find . -name '.DS_Store' -type f -delete || true"
run "find . -name '__MACOSX' -type d -prune -exec rm -rf {} + || true"
run "find . -name '._*' -type f -delete || true"

# 2) 删除缓存类目录（直接删）
run "find . -type d -name '__pycache__' -prune -exec rm -rf {} + || true"
run "find . -type d -name '.ipynb_checkpoints' -prune -exec rm -rf {} + || true"

# 3) 清理 LaTeX 编译产物（保留 src）
if [ -d "08_docs/Report/build" ]; then
  run "rm -rf '08_docs/Report/build' || true"
fi
run "find 08_docs -maxdepth 4 -type f \\( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.fls' -o -name '*.fdb_latexmk' -o -name '*.synctex.gz' -o -name '*.xdv' \\) -delete 2>/dev/null || true"

# 4) 处理重复与历史遗留输出（归档或删除）
# 你已经有规范输出：06_outputs/<RUN_TAG>/...
# 这些通常是重复或历史残留
move_or_delete "06_outputs/results"          # 旧路径聚合，容易产生嵌套 run
move_or_delete "99_tmp/run_conflict__"*      # 之前冲突备份（若存在）
move_or_delete "99_tmp/nested_run__"*        # 之前嵌套备份（若存在）

# 5) 定性图池很大，A 实验只需要你选中的那一小部分
# 如果你已经把选图放进 A_DIR/03_visualization/selected 或 07_figures/.../selected
# 那么 qual 大池可以归档
if [ -d "07_figures/$RUN_TAG/qual" ]; then
  move_or_delete "07_figures/$RUN_TAG/qual"
fi

# 6) 08_docs/paper_assets 如果你只以 A_DIR 为准，可归档它减少层级混乱
# 保留与否看你习惯。这里默认归档
if [ -d "08_docs/paper_assets" ]; then
  move_or_delete "08_docs/paper_assets"
fi

# 7) 根目录散落的临时脚本统一放 00_admin/scripts（不删除）
for f in ./*.py ./*.sh; do
  [ -f "$f" ] || continue
  bn="$(basename "$f")"
  if [ "$bn" != "00_admin" ]; then
    run "mv \"$f\" \"00_admin/scripts/\" || true"
  fi
done

# 8) 输出清理报告
mkdir -p 00_admin
{
  echo "timestamp: $TS"
  echo "RUN_DIR: $RUN_DIR"
  echo "RUN_TAG: $RUN_TAG"
  echo "PRUNE_DELETE: $PRUNE_DELETE"
  echo "ARCHIVE_DIR: $ARCH"
  echo
  echo "Top10 after prune:"
  du -sh ./* 2>/dev/null | sort -hr | head -n 10
  echo
  echo "Archive content:"
  du -sh "$ARCH"/* 2>/dev/null | sort -hr | head -n 30 || true
} > "00_admin/prune_report__${TS}.txt"

echo "DONE"
echo "report: 00_admin/prune_report__${TS}.txt"
echo "archive: $ARCH"
