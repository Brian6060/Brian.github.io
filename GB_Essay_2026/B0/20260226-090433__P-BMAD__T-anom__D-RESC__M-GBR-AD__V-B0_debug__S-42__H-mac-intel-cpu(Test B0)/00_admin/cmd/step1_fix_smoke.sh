#!/usr/bin/env bash
set -euo pipefail

RUN_DIR='/Users/dby051225/Desktop/GB Essay 2026/Test/20260226-090433__P-BMAD__T-anom__D-RESC__M-GBR-AD__V-B0_debug__S-42__H-mac-intel-cpu(Test B0)'
BMAD_DIR="$RUN_DIR/02_src/BMAD"

cd "$BMAD_DIR"

export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

# 只保留当前 BMAD 的 anomalib/src，彻底排毒
export PYTHONPATH="$BMAD_DIR/anomalib/src"

echo "RUN_DIR(env)=$RUN_DIR"
echo "BMAD_DIR=$BMAD_DIR"
echo "PYTHONPATH=$PYTHONPATH"

python - <<'PY'
import os, traceback
import anomalib
import anomalib.config.config as cc
from anomalib.config import get_configurable_parameters

print("anomalib.__file__ =", anomalib.__file__)
print("config.py =", cc.__file__)

try:
    get_configurable_parameters(model_name="gbr_ad")
except FileNotFoundError as e:
    print("EXPECTED CONFIG PATH FROM ERROR:")
    print(e)
PY

TEMPLATE="$BMAD_DIR/anomalib/src/anomalib/models/padim/config.yaml"
OUT_A="$BMAD_DIR/anomalib/src/anomalib/models/gbr_ad/config.yaml"
OUT_B="$BMAD_DIR/src/anomalib/models/gbr_ad/config.yaml"

mkdir -p "$(dirname "$OUT_A")" "$(dirname "$OUT_B")"

export TEMPLATE OUT_A OUT_B

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

python - <<'PY'
from anomalib.config import get_configurable_parameters
from anomalib.models import get_model

cfg = get_configurable_parameters(model_name="gbr_ad")
m = get_model(cfg)
print("OK factory", type(m))
PY
