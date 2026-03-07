set -euo pipefail

: "${RUN_DIR:?RUN_DIR not set}"
: "${BMAD_DIR:?BMAD_DIR not set}"
SEED="${SEED:-42}"

export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

cd "$BMAD_DIR/anomalib"
export PYTHONPATH="$PWD:$PWD/src:${PYTHONPATH:-}"
CONDA_ENV_PY="/Users/dby051225/opt/miniconda3/envs/gbessay_resc_rd4ad_cpu/bin/python"
if [ -x "$CONDA_ENV_PY" ]; then
  PYTHON_BIN="$CONDA_ENV_PY"
else
  PYTHON_BIN="python"
fi

mkdir -p "$RUN_DIR/05_logs/train" "$RUN_DIR/05_logs/test"
mkdir -p "$RUN_DIR/06_outputs/checkpoints" "$RUN_DIR/06_outputs/metrics"

if "$PYTHON_BIN" "$BMAD_DIR/anomalib/tools/test.py" --help 2>&1 | grep -q -- '--weights'; then
  WARG="--weights"
else
  WARG="--weight"
fi

pick_latest_file () {
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
patterns = os.environ["PATS"].split(";")
cands = []
for pat in patterns:
  cands += list(root.rglob(pat))
cands = [p for p in cands if p.is_file()]
if not cands:
  print("NO_FILE")
  raise SystemExit(0)
cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(str(cands[0]))
PY
}

run_one () {
  DATASET="$1"
  CFG="$2"
  RUN_PATH="$RUN_DIR/06_outputs/results/gbr_ad/$DATASET/run"

  echo
  echo "TRAIN $DATASET"
  "$PYTHON_BIN" "$BMAD_DIR/anomalib/tools/train.py" --model gbr_ad --config "$CFG" | tee "$RUN_DIR/05_logs/train/train__B0__${DATASET}__gbr_ad.txt"

  echo
  echo "PICK CKPT $DATASET"
  export ROOT="$RUN_PATH"
  export PATS="*.ckpt;model.ckpt;*.pt"
  CKPT="$(pick_latest_file)"
  if [ "$CKPT" = "NO_FILE" ]; then
    echo "NO ckpt found under $RUN_PATH"
    exit 2
  fi
  echo "CKPT=$CKPT"

  echo
  echo "TEST $DATASET"
  "$PYTHON_BIN" "$BMAD_DIR/anomalib/tools/test.py" --model gbr_ad --config "$CFG" "$WARG" "$CKPT" | tee "$RUN_DIR/05_logs/test/test__B0__${DATASET}__gbr_ad.txt"

  echo
  echo "PICK METRICS $DATASET"
  export ROOT="$RUN_PATH"
  export PATS="metrics.csv;*metrics*.csv;*.json;*.csv"
  METRIC="$(pick_latest_file)"
  if [ "$METRIC" = "NO_FILE" ] || [ -z "$METRIC" ]; then
    echo "No metrics file found. Fallback to parsing test log and generating CSV."
    OUTCSV="$RUN_DIR/06_outputs/metrics/metrics__B0__${DATASET}__seed${SEED}.csv"
    LOGTXT="$RUN_DIR/05_logs/test/test__B0__${DATASET}__gbr_ad.txt"
    export DATASET SEED LOGTXT OUTCSV
    "$PYTHON_BIN" - <<'PY'
import csv
import os

dataset = os.environ["DATASET"]
seed = os.environ["SEED"]
logtxt = os.environ["LOGTXT"]
outcsv = os.environ["OUTCSV"]

rows = []
seen = set()

with open(logtxt, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "│" not in line:
            continue
        parts = [p.strip() for p in line.split("│") if p.strip()]
        if len(parts) < 2:
            continue
        name = parts[0].replace(" ", "")
        if "_" in name:
            try:
                val = float(parts[1])
            except Exception:
                continue
            if name not in seen:
                rows.append((name, val))
                seen.add(name)

if not rows:
    raise SystemExit("Could not parse any metric rows from test log.")

with open(outcsv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "seed", "metric", "value"])
    for metric, value in rows:
        w.writerow([dataset, seed, metric, value])

print("WROTE", outcsv)
PY
    METRIC="$OUTCSV"
  fi
  echo "METRIC=$METRIC"

  cp -f "$CKPT" "$RUN_DIR/06_outputs/checkpoints/model__B0__${DATASET}__seed${SEED}.ckpt"

  EXT="${METRIC##*.}"
  METRIC_DEST="$RUN_DIR/06_outputs/metrics/metrics__B0__${DATASET}__seed${SEED}.${EXT}"
  if [ "$METRIC" != "$METRIC_DEST" ]; then
    cp -f "$METRIC" "$METRIC_DEST"
  fi

  echo "SAVED CKPT: $RUN_DIR/06_outputs/checkpoints/model__B0__${DATASET}__seed${SEED}.ckpt"
  echo "SAVED METR: $METRIC_DEST"
}

run_one "RESC" "$RUN_DIR/04_configs/RESC__GBR-AD__B0_debug.yaml"
run_one "OCT2017" "$RUN_DIR/04_configs/OCT2017__GBR-AD__B0_debug.yaml"

echo
echo "DONE Step3"
