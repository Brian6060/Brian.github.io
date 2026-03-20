from pathlib import Path
import re
import matplotlib.pyplot as plt

LOG_PATH = Path("/Users/dby051225/Desktop/VIS/U-Net/U-Net Experiment Reproduction/outputs/logs/train_weighted_1500iter.log")
OUT_PATH = Path("/Users/dby051225/Desktop/VIS/U-Net/U-Net Experiment Reproduction/reports/figures/training_curve.png")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

train_pat = re.compile(r"Iteration (\d+) .* loss = ([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
test_iter_pat = re.compile(r"Iteration (\d+), Testing net")
test_loss_pat = re.compile(r"Test net output #0: loss = ([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

train_iters, train_losses = [], []
val_iters, val_losses = [], []

pending_test_iter = None

with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m_test_iter = test_iter_pat.search(line)
        if m_test_iter:
            pending_test_iter = int(m_test_iter.group(1))
            continue

        m_test_loss = test_loss_pat.search(line)
        if m_test_loss and pending_test_iter is not None:
            val_iters.append(pending_test_iter)
            val_losses.append(float(m_test_loss.group(1)))
            pending_test_iter = None
            continue

        m_train = train_pat.search(line)
        if m_train:
            train_iters.append(int(m_train.group(1)))
            train_losses.append(float(m_train.group(2)))

if not train_iters:
    raise RuntimeError(f"No training loss found in log: {LOG_PATH}")

plt.figure(figsize=(8, 5))
plt.plot(train_iters, train_losses, linewidth=1.8, label="Train Loss")

if val_iters:
    plt.plot(val_iters, val_losses, marker="o", linewidth=1.6, label="Validation Loss")

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")

print(f"Saved figure to: {OUT_PATH}")
print(f"Number of training points: {len(train_iters)}")
print(f"Number of validation points: {len(val_iters)}")
