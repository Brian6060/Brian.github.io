set -euo pipefail

RUN_DIR="/Users/dby051225/Desktop/GB Essay 2026/Test/20260226-090433__P-BMAD__T-anom__D-RESC__M-GBR-AD__V-B0_debug__S-42__H-mac-intel-cpu(Test B0)"
BMAD_DIR="$RUN_DIR/02_src/BMAD"

cd "$BMAD_DIR"

export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

# 关键点: 只把当前 BMAD 的 anomalib/src 放进 PYTHONPATH
export PYTHONPATH="$BMAD_DIR/anomalib/src:$PYTHONPATH"

echo "PYTHONPATH=$PYTHONPATH"
echo "PWD=$(pwd)"

python - <<'PY'
import os, inspect, traceback
import anomalib
print("anomalib.__file__ =", anomalib.__file__)

from anomalib.config import get_configurable_parameters
import anomalib.config.config as cc
print("config.py =", cc.__file__)

# 先复现一次，打印它究竟在找哪个 config.yaml
try:
    get_configurable_parameters(model_name="gbr_ad")
except FileNotFoundError as e:
    print("EXPECTED CONFIG PATH FROM ERROR:")
    print(e)
PY

# 从一个已有模型复制模板 config
TEMPLATE=""
for p in \
  "$BMAD_DIR/anomalib/src/anomalib/models/padim/config.yaml" \
  "$BMAD_DIR/anomalib/src/anomalib/models/efficient_ad/config.yaml" \
  "$BMAD_DIR/anomalib/src/anomalib/models/reverse_distillation/config.yaml"
do
  if [ -f "$p" ]; then TEMPLATE="$p"; break; fi
done

if [ -z "$TEMPLATE" ]; then
  TEMPLATE="$(find "$BMAD_DIR/anomalib/src/anomalib/models" -maxdepth 2 -name config.yaml | head -n 1 || true)"
fi

if [ -z "$TEMPLATE" ]; then
  echo "FAIL: cannot find any template config.yaml under anomalib/models"
  exit 1
fi

echo "TEMPLATE=$TEMPLATE"

# 两个候选落点都写一份，防止工厂内部用不同根目录拼路径
OUT_A="$BMAD_DIR/anomalib/src/anomalib/models/gbr_ad/config.yaml"
OUT_B="$BMAD_DIR/src/anomalib/models/gbr_ad/config.yaml"

mkdir -p "$(dirname "$OUT_A")"
mkdir -p "$(dirname "$OUT_B")"

python - <<'PY'
import os
from omegaconf import OmegaConf

template = os.environ["TEMPLATE"]
out_a = os.environ["OUT_A"]
out_b = os.environ["OUT_B"]

cfg = OmegaConf.load(template)
if "model" not in cfg:
    cfg.model = {}
cfg.model.name = "gbr_ad"

OmegaConf.save(cfg, out_a)
OmegaConf.save(cfg, out_b)

print("written:", out_a)
print("written:", out_b)
PY

# 再跑一次真正的 smoke test
python - <<'PY'
import traceback
from anomalib.config import get_configurable_parameters
from anomalib.models import get_model

cfg = get_configurable_parameters(model_name="gbr_ad")
m = get_model(cfg)
print("OK factory", type(m))
PY
