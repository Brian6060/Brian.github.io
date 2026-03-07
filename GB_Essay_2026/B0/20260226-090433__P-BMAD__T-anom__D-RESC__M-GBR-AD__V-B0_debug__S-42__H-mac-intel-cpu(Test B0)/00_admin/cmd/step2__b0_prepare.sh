#!/usr/bin/env bash
set -euo pipefail

RUN_DIR='/Users/dby051225/Desktop/GB Essay 2026/Test/20260226-090433__P-BMAD__T-anom__D-RESC__M-GBR-AD__V-B0_debug__S-42__H-mac-intel-cpu(Test B0)'
BMAD_DIR="$RUN_DIR/02_src/BMAD"

export RUN_DIR
export BMAD_DIR

echo "[Step 2.0] Ensure folder structure"
mkdir -p \
  "$RUN_DIR/03_data" \
  "$RUN_DIR/04_configs" \
  "$RUN_DIR/05_logs" \
  "$RUN_DIR/06_outputs"

choose_first_existing_dir() {
  local selected=""
  local candidate=""
  for candidate in "$@"; do
    if [ -d "$candidate" ]; then
      selected="$candidate"
      break
    fi
  done
  if [ -z "$selected" ]; then
    return 1
  fi
  printf '%s\n' "$selected"
}

assert_min_files() {
  local dir="$1"
  local need="$2"
  local count
  count="$(find "$dir" -maxdepth 1 -type f | wc -l | tr -d ' ')"
  if [ "$count" -lt "$need" ]; then
    echo "ERROR: not enough files in $dir (have=$count, need=$need)" >&2
    exit 1
  fi
}

populate_stub_symlinks() {
  local src_dir="$1"
  local dst_dir="$2"
  local limit="$3"
  local linked=0
  local src_file=""

  rm -f "$dst_dir"/*
  while IFS= read -r src_file; do
    [ -n "$src_file" ] || continue
    ln -sfn "$src_file" "$dst_dir/$(basename "$src_file")"
    linked=$((linked + 1))
  done < <(find "$src_dir" -maxdepth 1 -type f | sort | head -n "$limit")

  echo "  linked $linked files: $src_dir -> $dst_dir"
}

echo "[Step 2.1] Locate real dataset roots and split leaves"
DATASETS_ROOT="$RUN_DIR/03_data/BMAD/datasets"
RESC_ROOT="$DATASETS_ROOT/RESC"
OCT_ROOT="$DATASETS_ROOT/OCT2017"

if [ ! -d "$RESC_ROOT" ] || [ ! -d "$OCT_ROOT" ]; then
  echo "WARN: expected dataset roots missing under $DATASETS_ROOT. Searching inside RUN_DIR..."
  find "$RUN_DIR" -type d \( -name "RESC" -o -name "OCT2017" -o -name "resc" -o -name "oct2017" \) | sed 's#^#  CAND: #'
  echo "ERROR: Could not use required dataset roots. Please fix dataset paths."
  exit 1
fi

RESC_TRAIN_GOOD="$(choose_first_existing_dir "$RESC_ROOT/train/good/data" "$RESC_ROOT/train/good")"
RESC_TEST_GOOD="$(choose_first_existing_dir "$RESC_ROOT/test/good/data" "$RESC_ROOT/test/good/img" "$RESC_ROOT/test/good")"
RESC_TEST_UNGOOD="$(choose_first_existing_dir "$RESC_ROOT/test/ungood/data" "$RESC_ROOT/test/ungood/img" "$RESC_ROOT/test/ungood")"
RESC_MASKS="$(choose_first_existing_dir "$RESC_ROOT/test/ungood/label")"

OCT_TRAIN_GOOD="$(choose_first_existing_dir "$OCT_ROOT/train/good/data" "$OCT_ROOT/train/good")"
OCT_TEST_GOOD="$(choose_first_existing_dir "$OCT_ROOT/test/good/data" "$OCT_ROOT/test/good/img" "$OCT_ROOT/test/good")"
OCT_TEST_UNGOOD="$(choose_first_existing_dir "$OCT_ROOT/test/ungood/data" "$OCT_ROOT/test/Ungood/img" "$OCT_ROOT/test/ungood" "$OCT_ROOT/test/Ungood")"

echo "  RESC_TRAIN_GOOD=$RESC_TRAIN_GOOD"
echo "  RESC_TEST_GOOD=$RESC_TEST_GOOD"
echo "  RESC_TEST_UNGOOD=$RESC_TEST_UNGOOD"
echo "  RESC_MASKS=$RESC_MASKS"
echo "  OCT_TRAIN_GOOD=$OCT_TRAIN_GOOD"
echo "  OCT_TEST_GOOD=$OCT_TEST_GOOD"
echo "  OCT_TEST_UNGOOD=$OCT_TEST_UNGOOD"

assert_min_files "$RESC_TRAIN_GOOD" 20
assert_min_files "$RESC_TEST_GOOD" 20
assert_min_files "$RESC_TEST_UNGOOD" 20
assert_min_files "$RESC_MASKS" 20
assert_min_files "$OCT_TRAIN_GOOD" 20
assert_min_files "$OCT_TEST_GOOD" 20
assert_min_files "$OCT_TEST_UNGOOD" 20

echo "[Step 2.2] Create stub roots"
RESC_STUB="$RUN_DIR/03_data/__b0_stub__/RESC"
OCT_STUB="$RUN_DIR/03_data/__b0_stub__/OCT2017"
mkdir -p \
  "$RESC_STUB/train/good/data" \
  "$RESC_STUB/test/good/data" \
  "$RESC_STUB/test/ungood/data" \
  "$RESC_STUB/test/ungood/label" \
  "$OCT_STUB/train/good/data" \
  "$OCT_STUB/test/good/data" \
  "$OCT_STUB/test/ungood/data"

echo "[Step 2.3] Populate stubs via deterministic symlinks (first 20 sorted files)"
populate_stub_symlinks "$RESC_TRAIN_GOOD" "$RESC_STUB/train/good/data" 20
populate_stub_symlinks "$RESC_TEST_GOOD" "$RESC_STUB/test/good/data" 20
populate_stub_symlinks "$RESC_TEST_UNGOOD" "$RESC_STUB/test/ungood/data" 20
populate_stub_symlinks "$RESC_MASKS" "$RESC_STUB/test/ungood/label" 20
populate_stub_symlinks "$OCT_TRAIN_GOOD" "$OCT_STUB/train/good/data" 20
populate_stub_symlinks "$OCT_TEST_GOOD" "$OCT_STUB/test/good/data" 20
populate_stub_symlinks "$OCT_TEST_UNGOOD" "$OCT_STUB/test/ungood/data" 20

echo "[Step 2.4] Generate configs using anomalib get_configurable_parameters"
CONDA_ENV_PY="/Users/dby051225/opt/miniconda3/envs/gbessay_resc_rd4ad_cpu/bin/python"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$CONDA_ENV_PY" ]; then
    PYTHON_BIN="$CONDA_ENV_PY"
  else
    PYTHON_BIN="python"
  fi
fi
echo "  PYTHON_BIN=$PYTHON_BIN"
export PYTHONPATH="$BMAD_DIR/anomalib/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" "$RUN_DIR/00_admin/cmd/step2__make_configs.py"

echo "[Step 2.5] Final checklist"
count_links() {
  find "$1" -maxdepth 1 -type l | wc -l | tr -d ' '
}

echo "Stub split counts:"
echo "  RESC train/good/data      : $(count_links "$RESC_STUB/train/good/data")"
echo "  RESC test/good/data       : $(count_links "$RESC_STUB/test/good/data")"
echo "  RESC test/ungood/data     : $(count_links "$RESC_STUB/test/ungood/data")"
echo "  RESC test/ungood/label    : $(count_links "$RESC_STUB/test/ungood/label")"
echo "  OCT2017 train/good/data   : $(count_links "$OCT_STUB/train/good/data")"
echo "  OCT2017 test/good/data    : $(count_links "$OCT_STUB/test/good/data")"
echo "  OCT2017 test/ungood/data  : $(count_links "$OCT_STUB/test/ungood/data")"

RESC_CFG="$RUN_DIR/04_configs/RESC__GBR-AD__B0_debug.yaml"
OCT_CFG="$RUN_DIR/04_configs/OCT2017__GBR-AD__B0_debug.yaml"
echo "YAML files:"
echo "  $RESC_CFG => $([ -f "$RESC_CFG" ] && echo YES || echo NO)"
echo "  $OCT_CFG  => $([ -f "$OCT_CFG" ] && echo YES || echo NO)"

echo "Done: Step 2 preparation complete."
