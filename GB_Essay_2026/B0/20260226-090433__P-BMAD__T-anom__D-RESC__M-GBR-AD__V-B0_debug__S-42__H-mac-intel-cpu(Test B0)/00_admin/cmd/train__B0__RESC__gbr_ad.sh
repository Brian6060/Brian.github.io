set -euo pipefail
RUN_DIR='/Users/dby051225/Desktop/GB Essay 2026/Test/20260226-090433__P-BMAD__T-anom__D-RESC__M-GBR-AD__V-B0_debug__S-42__H-mac-intel-cpu(Test B0)'
BMAD_DIR="$RUN_DIR/02_src/BMAD"
CFG="$RUN_DIR/04_configs/RESC__GBR-AD__B0_debug.yaml"

export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export PYTHONPATH="$BMAD_DIR/anomalib/src"

CONDA_ENV_PY="/Users/dby051225/opt/miniconda3/envs/gbessay_resc_rd4ad_cpu/bin/python"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$CONDA_ENV_PY" ]; then
    PYTHON_BIN="$CONDA_ENV_PY"
  else
    PYTHON_BIN="python"
  fi
fi
echo "PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" - <<'PYCHK'
import importlib.util, sys
missing = [m for m in ("omegaconf", "anomalib") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing python packages: {', '.join(missing)} (python={sys.executable})")
print(f"python deps ok: {sys.executable}")
PYCHK

echo "PRE-FLIGHT CONFIG DUMP"
"$PYTHON_BIN" - <<'PYCFG'
from omegaconf import OmegaConf
cfg = OmegaConf.load("/Users/dby051225/Desktop/GB Essay 2026/Test/20260226-090433__P-BMAD__T-anom__D-RESC__M-GBR-AD__V-B0_debug__S-42__H-mac-intel-cpu(Test B0)/04_configs/RESC__GBR-AD__B0_debug.yaml")
print("CFG file =", "/Users/dby051225/Desktop/GB Essay 2026/Test/20260226-090433__P-BMAD__T-anom__D-RESC__M-GBR-AD__V-B0_debug__S-42__H-mac-intel-cpu(Test B0)/04_configs/RESC__GBR-AD__B0_debug.yaml")
print("dataset.root =", getattr(cfg.dataset, "root", None))
print("dataset.normal_dir =", getattr(cfg.dataset, "normal_dir", None))
print("dataset.abnormal_dir =", getattr(cfg.dataset, "abnormal_dir", None))
print("dataset.mask_dir =", getattr(cfg.dataset, "mask_dir", None))
print("dataset.extensions =", getattr(cfg.dataset, "extensions", None))
PYCFG

"$PYTHON_BIN" "$BMAD_DIR/anomalib/tools/train.py" --model gbr_ad --config "$CFG"
