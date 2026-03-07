set -euo pipefail

RUN_DIR="/Users/dby051225/Desktop/GB Essay 2026/Test/20260226-090433__P-BMAD__T-anom__D-RESC__M-GBR-AD__V-B0_debug__S-42__H-mac-intel-cpu(Test B0)"
BMAD_DIR="$RUN_DIR/02_src/BMAD"

echo "RUN_DIR=$RUN_DIR" | tee "$RUN_DIR/00_admin/snapshots/run_dir.txt"
date | tee "$RUN_DIR/00_admin/snapshots/date.txt"

python -V | tee "$RUN_DIR/00_admin/snapshots/python_version.txt"
python -c "import platform; print(platform.platform())" | tee "$RUN_DIR/00_admin/snapshots/platform.txt"
python -c "import torch; print(torch.__version__)" | tee "$RUN_DIR/00_admin/snapshots/torch_version.txt"

python -c "import numpy,scipy; print('numpy', numpy.__version__); print('scipy', scipy.__version__)" \
  | tee "$RUN_DIR/00_admin/snapshots/numpy_scipy.txt"

python -c "import hydra,omegaconf; print('hydra', hydra.__version__); print('omegaconf', omegaconf.__version__)" \
  | tee "$RUN_DIR/00_admin/snapshots/hydra_omegaconf.txt"

pip freeze > "$RUN_DIR/00_admin/snapshots/pip_freeze.txt" || true

if [ -d "$BMAD_DIR/.git" ]; then
  (cd "$BMAD_DIR" && git rev-parse HEAD) | tee "$RUN_DIR/00_admin/snapshots/git_commit__BMAD.txt" || true
  (cd "$BMAD_DIR" && git status --porcelain) > "$RUN_DIR/00_admin/snapshots/git_status__BMAD.txt" || true
fi
