from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


BASE_DIR = Path("/Users/brian/Desktop/VCL318/Swin-ViT/ViTUnet vs SwinUnet")
FINAL_OUTPUT_DIR = BASE_DIR / "Final Output"

VIT_RUN_DIR = BASE_DIR / "ViTUnet" / "runs" / "vitunet_kvasir_seg_pretrained_bs8"
SWIN_RUN_DIR = BASE_DIR / "SwinUnet" / "runs" / "swinunet_kvasir_seg_pretrained_bs8"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def try_load_json(path: Path) -> Optional[dict]:
    return load_json(path) if path.exists() else None


def normalize_metric_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "filename" in df.columns and "stem" not in df.columns:
        df["stem"] = df["filename"].astype(str).apply(lambda x: Path(x).stem)
    elif "stem" in df.columns:
        df["stem"] = df["stem"].astype(str).apply(lambda x: Path(x).stem)
    else:
        raise ValueError("Per-image metric CSV must contain either 'filename' or 'stem'.")

    for col in ["iou", "dice", "ap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["stem", "iou", "dice", "ap"]].dropna().sort_values("stem").reset_index(drop=True)


def load_train_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "val_ap"]].copy()


def pick_overlay_path(run_dir: Path, stem: str) -> Optional[Path]:
    for p in [
        run_dir / "test_overlays" / f"{stem}_overlay.png",
        run_dir / "inference_full" / "overlays" / f"{stem}_overlay.png",
    ]:
        if p.exists():
            return p
    return None


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_run_bundle(run_dir: Path) -> dict:
    metrics_dir = run_dir / "metrics"
    return {
        "run_dir": run_dir,
        "summary": load_json(metrics_dir / "test_metrics_summary.json"),
        "train_summary": try_load_json(metrics_dir / "train_summary.json"),
        "exp_config": try_load_json(metrics_dir / "experiment_config.json"),
        "per_image": normalize_metric_df(pd.read_csv(metrics_dir / "test_metrics_per_image.csv")),
        "train_log": load_train_log(run_dir / "logs" / "train_log.csv"),
    }


def plot_summary_bars(vit_summary: dict, swin_summary: dict, out_dir: Path) -> None:
    metrics = ["mean_iou", "mean_dice", "mean_ap"]
    labels = ["IoU", "Dice", "AP"]
    vit_vals = [float(vit_summary[m]) for m in metrics]
    swin_vals = [float(swin_summary[m]) for m in metrics]
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(8, 5), dpi=220)
    bars1 = plt.bar(x - width / 2, vit_vals, width, label="ViTUNet")
    bars2 = plt.bar(x + width / 2, swin_vals, width, label="SwinUNet")
    plt.xticks(x, labels)
    plt.ylim(0, max(vit_vals + swin_vals) * 1.18)
    plt.ylabel("Score")
    plt.title("Test Metrics Comparison")
    plt.legend()
    for bars in [bars1, bars2]:
        for b in bars:
            h = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2, h + 0.003, f"{h:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "fig01_summary_bar.png")
    plt.close()


def plot_train_curves(vit_log: pd.DataFrame, swin_log: pd.DataFrame, out_dir: Path) -> None:
    pairs = [
        ("train_loss", "Training Loss"),
        ("val_loss", "Validation Loss"),
        ("val_iou", "Validation IoU"),
        ("val_dice", "Validation Dice"),
        ("val_ap", "Validation AP"),
    ]
    for col, title in pairs:
        plt.figure(figsize=(8, 5), dpi=220)
        plt.plot(vit_log["epoch"], vit_log[col], label="ViTUNet")
        plt.plot(swin_log["epoch"], swin_log[col], label="SwinUNet")
        plt.xlabel("Epoch")
        plt.ylabel(col)
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"curve_{col}.png")
        plt.close()


def plot_boxplots(merged: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=220)
    metric_pairs = [
        ("iou_vit", "iou_swin", "IoU"),
        ("dice_vit", "dice_swin", "Dice"),
        ("ap_vit", "ap_swin", "AP"),
    ]
    for ax, (cv, cs, title) in zip(axes, metric_pairs):
        ax.boxplot([merged[cv].values, merged[cs].values], labels=["ViT", "Swin"], showmeans=True)
        ax.set_title(f"Per-image {title}")
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "fig02_boxplots.png")
    plt.close()


def plot_scatter_compare(merged: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), dpi=220)
    metric_pairs = [
        ("iou_vit", "iou_swin", "IoU"),
        ("dice_vit", "dice_swin", "Dice"),
        ("ap_vit", "ap_swin", "AP"),
    ]
    for ax, (cv, cs, title) in zip(axes, metric_pairs):
        x = merged[cv].values
        y = merged[cs].values
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.scatter(x, y, s=16, alpha=0.7)
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel(f"ViT {title}")
        ax.set_ylabel(f"Swin {title}")
        ax.set_title(f"Swin vs ViT ({title})")
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "fig03_scatter_swin_vs_vit.png")
    plt.close()


def plot_hist_diffs(merged: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), dpi=220)
    metric_pairs = [
        ("diff_iou", "IoU"),
        ("diff_dice", "Dice"),
        ("diff_ap", "AP"),
    ]
    for ax, (col, title) in zip(axes, metric_pairs):
        vals = merged[col].values
        ax.hist(vals, bins=20)
        ax.axvline(vals.mean(), linestyle="--", label=f"mean={vals.mean():.4f}")
        ax.set_title(f"Swin - ViT ({title})")
        ax.legend()
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "fig04_hist_metric_diffs.png")
    plt.close()


def plot_ecdf_dice(merged: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7.5, 5), dpi=220)
    for col, label in [("dice_vit", "ViTUNet"), ("dice_swin", "SwinUNet")]:
        x = np.sort(merged[col].values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=label)
    plt.xlabel("Per-image Dice")
    plt.ylabel("ECDF")
    plt.title("Empirical CDF of Dice")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig05_ecdf_dice.png")
    plt.close()


def plot_ranked_dice(merged: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8.5, 5), dpi=220)
    plt.plot(np.sort(merged["dice_vit"].values), label="ViTUNet")
    plt.plot(np.sort(merged["dice_swin"].values), label="SwinUNet")
    plt.xlabel("Sorted Test Cases")
    plt.ylabel("Dice")
    plt.title("Ranked Per-image Dice")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig06_ranked_dice.png")
    plt.close()


def plot_case_strip(merged: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(10, 5), dpi=220)
    x = np.arange(len(merged))
    plt.scatter(x, merged["dice_vit"], s=10, alpha=0.7, label="ViT Dice")
    plt.scatter(x, merged["dice_swin"], s=10, alpha=0.7, label="Swin Dice")
    plt.xlabel("Case Index")
    plt.ylabel("Dice")
    plt.title("Per-case Dice Strip Plot")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "fig07_case_strip_dice.png")
    plt.close()


def wrap(s: Any) -> str:
    return "\n".join(textwrap.wrap(str(s), width=28))


def plot_config_table(vit_cfg: dict, swin_cfg: dict, vit_train: dict, swin_train: dict, out_dir: Path) -> None:
    rows = [
        ["backbone", vit_cfg.get("backbone"), swin_cfg.get("backbone")],
        ["pretrained", vit_cfg.get("pretrained"), swin_cfg.get("pretrained")],
        ["image_size", vit_cfg.get("image_size"), swin_cfg.get("image_size")],
        ["batch_size", vit_cfg.get("batch_size"), swin_cfg.get("batch_size")],
        ["epochs", vit_cfg.get("epochs"), swin_cfg.get("epochs")],
        ["optimizer", vit_cfg.get("optimizer"), swin_cfg.get("optimizer")],
        ["lr", vit_cfg.get("lr"), swin_cfg.get("lr")],
        ["weight_decay", vit_cfg.get("weight_decay"), swin_cfg.get("weight_decay")],
        ["loss", wrap(vit_cfg.get("loss")), wrap(swin_cfg.get("loss"))],
        ["best_epoch", vit_train.get("best_epoch") if vit_train else None, swin_train.get("best_epoch") if swin_train else None],
        ["best_dice", vit_train.get("best_dice") if vit_train else None, swin_train.get("best_dice") if swin_train else None],
    ]
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=220)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Field", "ViTUNet", "SwinUNet"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title("Experiment Configuration Comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "fig08_config_table.png")
    plt.close()


def create_top_case_csv(merged: pd.DataFrame, out_dir: Path) -> None:
    vit_better = merged.sort_values("diff_dice").head(15).copy()
    swin_better = merged.sort_values("diff_dice", ascending=False).head(15).copy()
    vit_better["winner"] = "ViT"
    swin_better["winner"] = "Swin"
    pd.concat([vit_better, swin_better], axis=0, ignore_index=True).to_csv(out_dir / "top_cases_comparison.csv", index=False)


def create_metrics_table_csv(vit_summary: dict, swin_summary: dict, out_dir: Path) -> None:
    pd.DataFrame(
        [
            ["ViTUNet", vit_summary["mean_iou"], vit_summary["mean_dice"], vit_summary["mean_ap"]],
            ["SwinUNet", swin_summary["mean_iou"], swin_summary["mean_dice"], swin_summary["mean_ap"]],
        ],
        columns=["model", "mean_iou", "mean_dice", "mean_ap"],
    ).to_csv(out_dir / "final_metrics_table.csv", index=False)


def create_case_gallery(run_dir_vit: Path, run_dir_swin: Path, merged: pd.DataFrame, out_dir: Path) -> None:
    top_swin = merged.sort_values("diff_dice", ascending=False).head(3)
    top_vit = merged.sort_values("diff_dice", ascending=True).head(3)
    selected = pd.concat([top_swin, top_vit], axis=0)
    rows = []
    for _, r in selected.iterrows():
        stem = r["stem"]
        p_vit = pick_overlay_path(run_dir_vit, stem)
        p_swin = pick_overlay_path(run_dir_swin, stem)
        if p_vit and p_swin:
            rows.append((stem, p_vit, p_swin, r["dice_vit"], r["dice_swin"]))
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(8.5, 3.8 * len(rows)), dpi=220)
    if len(rows) == 1:
        axes = np.array([axes])
    for i, (stem, p_vit, p_swin, dv, ds) in enumerate(rows):
        axes[i, 0].imshow(np.array(Image.open(p_vit).convert("RGB")))
        axes[i, 0].set_title(f"ViT | {stem}\nDice={dv:.4f}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(np.array(Image.open(p_swin).convert("RGB")))
        axes[i, 1].set_title(f"Swin | {stem}\nDice={ds:.4f}")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "fig09_case_gallery.png")
    plt.close()


def create_markdown_report(vit_bundle: dict, swin_bundle: dict, merged: pd.DataFrame, out_dir: Path) -> None:
    vit_s = vit_bundle["summary"]
    swin_s = swin_bundle["summary"]
    diff_iou = float(swin_s["mean_iou"]) - float(vit_s["mean_iou"])
    diff_dice = float(swin_s["mean_dice"]) - float(vit_s["mean_dice"])
    diff_ap = float(swin_s["mean_ap"]) - float(vit_s["mean_ap"])
    text = f"""# ViTUNet vs SwinUNet Final Comparison

## Final test summary
- ViTUNet: IoU={vit_s['mean_iou']:.6f}, Dice={vit_s['mean_dice']:.6f}, AP={vit_s['mean_ap']:.6f}
- SwinUNet: IoU={swin_s['mean_iou']:.6f}, Dice={swin_s['mean_dice']:.6f}, AP={swin_s['mean_ap']:.6f}

## Gaps (Swin - ViT)
- IoU gap: {diff_iou:.6f}
- Dice gap: {diff_dice:.6f}
- AP gap: {diff_ap:.6f}

## Case-level Dice wins
- Swin better: {(merged['diff_dice'] > 0).sum()}
- ViT better: {(merged['diff_dice'] < 0).sum()}
- Tie: {(merged['diff_dice'] == 0).sum()}

## Recommended figures
- fig01_summary_bar.png
- curve_val_dice.png
- fig02_boxplots.png
- fig03_scatter_swin_vs_vit.png
- fig04_hist_metric_diffs.png
- fig09_case_gallery.png
"""
    save_text(out_dir / "final_report.md", text)


def main() -> None:
    ensure_dir(FINAL_OUTPUT_DIR)
    plots_dir = FINAL_OUTPUT_DIR / "plots"
    ensure_dir(plots_dir)

    vit_bundle = load_run_bundle(VIT_RUN_DIR)
    swin_bundle = load_run_bundle(SWIN_RUN_DIR)

    vit_df = vit_bundle["per_image"].rename(columns={"iou": "iou_vit", "dice": "dice_vit", "ap": "ap_vit"})
    swin_df = swin_bundle["per_image"].rename(columns={"iou": "iou_swin", "dice": "dice_swin", "ap": "ap_swin"})

    merged = vit_df.merge(swin_df, on="stem", how="inner")
    merged["diff_iou"] = merged["iou_swin"] - merged["iou_vit"]
    merged["diff_dice"] = merged["dice_swin"] - merged["dice_vit"]
    merged["diff_ap"] = merged["ap_swin"] - merged["ap_vit"]
    merged = merged.sort_values("stem").reset_index(drop=True)

    merged.to_csv(FINAL_OUTPUT_DIR / "merged_per_image_metrics.csv", index=False)

    final_summary = {
        "vit_summary": vit_bundle["summary"],
        "swin_summary": swin_bundle["summary"],
        "num_merged_cases": int(len(merged)),
        "mean_diff_iou_swin_minus_vit": round(float(merged["diff_iou"].mean()), 6),
        "mean_diff_dice_swin_minus_vit": round(float(merged["diff_dice"].mean()), 6),
        "mean_diff_ap_swin_minus_vit": round(float(merged["diff_ap"].mean()), 6),
        "swin_better_dice_cases": int((merged["diff_dice"] > 0).sum()),
        "vit_better_dice_cases": int((merged["diff_dice"] < 0).sum()),
    }
    save_json(FINAL_OUTPUT_DIR / "final_summary.json", final_summary)

    create_metrics_table_csv(vit_bundle["summary"], swin_bundle["summary"], FINAL_OUTPUT_DIR)
    create_top_case_csv(merged, FINAL_OUTPUT_DIR)

    plot_summary_bars(vit_bundle["summary"], swin_bundle["summary"], plots_dir)
    plot_train_curves(vit_bundle["train_log"], swin_bundle["train_log"], plots_dir)
    plot_boxplots(merged, plots_dir)
    plot_scatter_compare(merged, plots_dir)
    plot_hist_diffs(merged, plots_dir)
    plot_ecdf_dice(merged, plots_dir)
    plot_ranked_dice(merged, plots_dir)
    plot_case_strip(merged, plots_dir)
    plot_config_table(vit_bundle["exp_config"] or {}, swin_bundle["exp_config"] or {}, vit_bundle["train_summary"] or {}, swin_bundle["train_summary"] or {}, plots_dir)
    create_case_gallery(VIT_RUN_DIR, SWIN_RUN_DIR, merged, plots_dir)
    create_markdown_report(vit_bundle, swin_bundle, merged, FINAL_OUTPUT_DIR)

    print("Done. Outputs saved to:", FINAL_OUTPUT_DIR)


if __name__ == "__main__":
    main()
