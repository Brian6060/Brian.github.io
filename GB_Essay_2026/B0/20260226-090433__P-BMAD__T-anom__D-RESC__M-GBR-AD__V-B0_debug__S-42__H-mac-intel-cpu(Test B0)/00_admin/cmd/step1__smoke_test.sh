set -euo pipefail

RUN_DIR="/Users/dby051225/Desktop/GB Essay 2026/Test/20260226-090433__P-BMAD__T-anom__D-RESC__M-GBR-AD__V-B0_debug__S-42__H-mac-intel-cpu(Test B0)"
BMAD_DIR="$RUN_DIR/02_src/BMAD"

cd "$BMAD_DIR"

export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

PY_ADD="$PWD"
if [ -d "$PWD/anomalib/src" ]; then
  PY_ADD="$PY_ADD:$PWD/anomalib/src"
fi
if [ -d "$PWD/src" ]; then
  PY_ADD="$PY_ADD:$PWD/src"
fi
export PYTHONPATH="$PY_ADD:$PYTHONPATH"

python - <<'PY'
import sys, traceback

def die(msg):
    print(msg)
    sys.exit(1)

try:
    from anomalib.config import get_configurable_parameters
except Exception:
    traceback.print_exc()
    die("FAIL import anomalib.config.get_configurable_parameters")

try:
    from anomalib.models import get_model
except Exception:
    traceback.print_exc()
    die("FAIL import anomalib.models.get_model")

try:
    cfg = get_configurable_parameters(model_name="gbr_ad")
    model = get_model(cfg)
except Exception:
    traceback.print_exc()
    die("FAIL model factory for gbr_ad")

print("OK factory", type(model))
PY
