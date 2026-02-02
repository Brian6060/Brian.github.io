import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_exposure_curves(
    x,
    alpha_model, beta_model, gamma_model,
    alpha_human, beta_human, gamma_human,
    title="Exposure of Occupations to GPTs",
    xlabel="Percentage of Exposure",
    ylabel="Percentage of Occupations at or Above",
    outpath="fig_exposure_of_occupations_to_gpts.pdf"
):
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)

    # --- Shaded bands (alpha -> gamma) for model and human ---
    ax.fill_between(x, alpha_model, gamma_model, color="red", alpha=0.18, linewidth=0)
    ax.fill_between(x, alpha_human, gamma_human, color="blue", alpha=0.18, linewidth=0)

    # --- Lines: model (red) ---
    ax.plot(x, alpha_model, color="red", marker="o", linewidth=2.0, label="α (model)")
    ax.plot(x, beta_model,  color="red", marker="^", linestyle="--", linewidth=2.0, label="β (model)")
    ax.plot(x, gamma_model, color="red", marker="s", linewidth=2.0, label="γ (model)")

    # --- Lines: human (blue) ---
    ax.plot(x, alpha_human, color="blue", marker="o", linewidth=2.0, label="α (human)")
    ax.plot(x, beta_human,  color="blue", marker="^", linestyle="--", linewidth=2.0, label="β (human)")
    ax.plot(x, gamma_human, color="blue", marker="s", linewidth=2.0, label="γ (human)")

    # --- Axes / grid / legend ---
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()


# =========================
# Option A) 从 CSV 读数据（推荐）
# CSV 列名建议：
# x, alpha_model, beta_model, gamma_model, alpha_human, beta_human, gamma_human
# =========================
def plot_from_csv(csv_path, outpath="fig_exposure_of_occupations_to_gpts.pdf"):
    df = pd.read_csv(csv_path)
    plot_exposure_curves(
        x=df["x"].to_numpy(),
        alpha_model=df["alpha_model"].to_numpy(),
        beta_model=df["beta_model"].to_numpy(),
        gamma_model=df["gamma_model"].to_numpy(),
        alpha_human=df["alpha_human"].to_numpy(),
        beta_human=df["beta_human"].to_numpy(),
        gamma_human=df["gamma_human"].to_numpy(),
        outpath=outpath
    )


# =========================
# Option B) 直接在代码里填数组
# （把下面这些示例数组替换成你自己的结果）
# =========================
if __name__ == "__main__":
    x = np.linspace(0, 1, 21)  # 0.00, 0.05, ..., 1.00

    # ---- 示例：请替换为你自己的6条曲线数据（长度需与 x 一致）----
    alpha_model = np.array([1.00,0.72,0.49,0.33,0.22,0.15,0.10,0.08,0.06,0.05,0.04,0.035,0.030,0.025,0.020,0.018,0.015,0.012,0.010,0.008,0.006])
    beta_model  = np.array([1.00,0.90,0.82,0.74,0.68,0.63,0.56,0.50,0.45,0.38,0.28,0.15,0.10,0.07,0.05,0.04,0.03,0.02,0.015,0.010,0.008])
    gamma_model = np.array([1.00,0.92,0.86,0.80,0.74,0.70,0.66,0.63,0.61,0.58,0.56,0.53,0.50,0.46,0.43,0.40,0.35,0.28,0.22,0.14,0.09])

    alpha_human = np.array([1.00,0.65,0.53,0.41,0.31,0.22,0.15,0.09,0.05,0.03,0.02,0.015,0.012,0.010,0.008,0.006,0.005,0.004,0.003,0.002,0.001])
    beta_human  = np.array([1.00,0.88,0.76,0.68,0.62,0.57,0.50,0.43,0.35,0.28,0.22,0.13,0.07,0.05,0.035,0.025,0.018,0.012,0.008,0.005,0.003])
    gamma_human = np.array([1.00,0.90,0.84,0.78,0.72,0.68,0.64,0.60,0.56,0.51,0.47,0.42,0.37,0.31,0.26,0.22,0.17,0.11,0.07,0.03,0.01])

    plot_exposure_curves(
        x,
        alpha_model, beta_model, gamma_model,
        alpha_human, beta_human, gamma_human,
        outpath="fig_exposure_of_occupations_to_gpts.pdf"
    )
