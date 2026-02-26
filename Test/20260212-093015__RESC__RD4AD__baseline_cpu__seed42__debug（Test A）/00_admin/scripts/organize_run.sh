#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
RUN_DIR="${RUN_DIR:-$PWD}"

if [ ! -d "$RUN_DIR" ]; then
  echo "RUN_DIR not found: $RUN_DIR"
  exit 1
fi

cd "$RUN_DIR"

RUN_TAG="${RUN_TAG:-$(basename "$RUN_DIR")}"

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf "[DRY] %s\n" "$*"
  else
    eval "$@"
  fi
}

ensure_dir() {
  run "mkdir -p '$1'"
}

mv_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "$src" ]; then
    ensure_dir "$(dirname "$dst")"
    run "mv '$src' '$dst'"
  fi
}

mv_glob_into_dir() {
  local pattern="$1"
  local dst_dir="$2"
  ensure_dir "$dst_dir"
  shopt -s nullglob
  local files=( $pattern )
  shopt -u nullglob
  if [ "${#files[@]}" -gt 0 ]; then
    for f in "${files[@]}"; do
      run "mv '$f' '$dst_dir/'"
    done
  fi
}

ensure_dir "00_admin"
ensure_dir "01_env"
ensure_dir "02_src"
ensure_dir "03_data"
ensure_dir "04_configs/$RUN_TAG"
ensure_dir "05_logs/$RUN_TAG/lightning_logs"
ensure_dir "06_outputs/$RUN_TAG/checkpoints"
ensure_dir "06_outputs/$RUN_TAG/metrics"
ensure_dir "06_outputs/$RUN_TAG/results_engine"
ensure_dir "06_outputs/$RUN_TAG/images"
ensure_dir "07_figures/$RUN_TAG/qual"
ensure_dir "07_figures/$RUN_TAG/paper_ready"
ensure_dir "08_docs"
ensure_dir "99_tmp"

ts="$(date +%Y%m%d-%H%M%S)"
ensure_dir "00_admin"
run "find . -maxdepth 6 -print > '00_admin/filetree_before__${ts}.txt'"

if [ -d "03_data" ]; then
  run "find '03_data' -name '.DS_Store' -type f -delete || true"
  run "find '03_data' -name '__MACOSX' -type d -prune -exec rm -rf {} + || true"
  run "find '03_data' -name '._*' -type f -delete || true"
fi

mv_glob_into_dir "04_configs/*.yaml" "04_configs/$RUN_TAG"
mv_glob_into_dir "04_configs/*.yml" "04_configs/$RUN_TAG"
mv_glob_into_dir "04_configs/*.json" "04_configs/$RUN_TAG"
mv_glob_into_dir "04_configs/*.patch" "04_configs/$RUN_TAG"

mv_glob_into_dir "05_logs/*.log" "05_logs/$RUN_TAG"
mv_glob_into_dir "05_logs/*.txt" "05_logs/$RUN_TAG"

if [ -d "06_outputs/checkpoints" ]; then
  shopt -s nullglob
  ckpts=( "06_outputs/checkpoints/"* )
  shopt -u nullglob
  if [ "${#ckpts[@]}" -gt 0 ]; then
    ensure_dir "06_outputs/$RUN_TAG/checkpoints"
    for f in "${ckpts[@]}"; do
      run "mv '$f' '06_outputs/$RUN_TAG/checkpoints/'"
    done
  fi
fi

if [ -d "06_outputs/metrics" ]; then
  shopt -s nullglob
  ms=( "06_outputs/metrics/"* )
  shopt -u nullglob
  if [ "${#ms[@]}" -gt 0 ]; then
    ensure_dir "06_outputs/$RUN_TAG/metrics"
    for f in "${ms[@]}"; do
      run "mv '$f' '06_outputs/$RUN_TAG/metrics/'"
    done
  fi
fi

if [ -d "03_data/BMAD/datasets/RESC/reverse_distillation/RESC/run" ]; then
  ensure_dir "06_outputs/$RUN_TAG/results_engine"
  if [ ! -d "06_outputs/$RUN_TAG/results_engine/reverse_distillation__RESC__run" ]; then
    run "mv '03_data/BMAD/datasets/RESC/reverse_distillation/RESC/run' '06_outputs/$RUN_TAG/results_engine/reverse_distillation__RESC__run'"
  else
    run "mv '03_data/BMAD/datasets/RESC/reverse_distillation/RESC/run' '99_tmp/run_conflict__${ts}'"
  fi

  run "rm -f '03_data/BMAD/datasets/RESC/reverse_distillation/RESC/run' || true"
  run "ln -s \"$(pwd)/06_outputs/$RUN_TAG/results_engine/reverse_distillation__RESC__run\" '03_data/BMAD/datasets/RESC/reverse_distillation/RESC/run'"
fi

if [ -d "06_outputs/results/reverse_distillation/RESC/run" ]; then
  ensure_dir "06_outputs/results/reverse_distillation/RESC"
  run "mv '06_outputs/results/reverse_distillation/RESC/run' '06_outputs/results/reverse_distillation/RESC/$RUN_TAG'"
fi

if [ -d "07_figures/RESC_RD4AD_seed42_debug" ]; then
  ensure_dir "07_figures/$RUN_TAG/qual"
  run "mv '07_figures/RESC_RD4AD_seed42_debug' '07_figures/$RUN_TAG/qual/'"
fi

if [ -d "Report" ]; then
  ensure_dir "08_docs"
  run "mv 'Report' '08_docs/Report'"
  ensure_dir "08_docs/Report/src"
  ensure_dir "08_docs/Report/build"

  if [ -f "08_docs/Report/Report.tex" ]; then
    run "mv '08_docs/Report/Report.tex' '08_docs/Report/src/Report.tex'"
  fi

  shopt -s nullglob
  auxs=( "08_docs/Report/"*.aux "08_docs/Report/"*.log "08_docs/Report/"*.out "08_docs/Report/"*.fls "08_docs/Report/"*.fdb_latexmk "08_docs/Report/"*.synctex.gz "08_docs/Report/"*.xdv )
  shopt -u nullglob
  if [ "${#auxs[@]}" -gt 0 ]; then
    for f in "${auxs[@]}"; do
      run "mv '$f' '08_docs/Report/build/'"
    done
  fi
fi

run "find . -maxdepth 6 -print > '00_admin/filetree_after__${ts}.txt'"

echo "RUN_DIR: $RUN_DIR"
echo "RUN_TAG: $RUN_TAG"
echo "DRY_RUN: $DRY_RUN"
echo "DONE"
