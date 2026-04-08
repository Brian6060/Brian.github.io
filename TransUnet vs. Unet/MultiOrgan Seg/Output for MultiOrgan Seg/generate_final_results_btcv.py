#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


UNET_FINAL_ROOT = Path(
    "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/U-Net for MultiOrgan Seg/output/final_output"
)
TRANS_FINAL_ROOT = Path(
    "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/TransUnet for MultiOrgan Seg/output/final_output"
)
UNET_TRAIN_LOG = Path(
    "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/U-Net for MultiOrgan Seg/output/train_output/train_log.csv"
)
TRANS_TRAIN_LOG = Path(
    "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/TransUnet for MultiOrgan Seg/output/train_output/train_log.csv"
)
OUTPUT_ROOT = Path(
    "/Users/brian/Desktop/VCL318/TransU-Net/TransUnet vs. Unet/Output for MultiOrgan Seg/final_comparison"
)


def ensure_dirs() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT


def safe_read_csv(path: Path, name: str = "") -> Optional[pd.DataFrame]:
    if not path.exists():
        warnings.warn(f"Missing CSV file: {path}" if not name else f"Missing {name}: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Failed to read CSV {path}: {exc}")
        return None


def safe_read_json(path: Path, name: str = "") -> Optional[Dict]:
    if not path.exists():
        warnings.warn(f"Missing JSON file: {path}" if not name else f"Missing {name}: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        warnings.warn(f"Failed to read JSON {path}: {exc}")
        return None


def _read_summary_csv(path: Path) -> Optional[pd.DataFrame]:
    df = safe_read_csv(path, "final_summary.csv")
    if df is None or df.empty:
        warnings.warn(f"Empty or unreadable final summary: {path}")
        return None
    return df


def load_model_results(model_tag: str, final_root: Path, train_log_path: Path) -> Dict[str, object]:
    summary_csv = _read_summary_csv(final_root / "final_summary.csv")
    summary_json = safe_read_json(final_root / "final_summary.json", "final_summary.json")
    metrics_per_class = safe_read_csv(final_root / "metrics_per_class.csv", "metrics_per_class.csv")
    metrics_per_sample = safe_read_csv(final_root / "metrics_per_sample.csv", "metrics_per_sample.csv")
    inference_records = safe_read_csv(final_root / "inference_records.csv", "inference_records.csv")
    train_log = safe_read_csv(train_log_path, "train_log.csv")

    model_name = model_tag
    if summary_csv is not None and "model_name" in summary_csv.columns and not summary_csv.empty:
        model_name = str(summary_csv.iloc[0]["model_name"])
    elif summary_json is not None:
        model_name = str(summary_json.get("model_name", model_name))

    if metrics_per_sample is not None:
        for column in ["mean_dice", "mean_iou"]:
            if column in metrics_per_sample.columns:
                metrics_per_sample[column] = pd.to_numeric(metrics_per_sample[column], errors="coerce")

    if metrics_per_class is not None:
        for column in ["class_id", "mean_dice", "mean_iou", "support_samples"]:
            if column in metrics_per_class.columns:
                metrics_per_class[column] = pd.to_numeric(metrics_per_class[column], errors="coerce")

    overlay_map: Dict[tuple[str, str], str] = {}
    if inference_records is not None and {"case_id", "slice_id", "overlay_path"}.issubset(inference_records.columns):
        for _, row in inference_records.iterrows():
            key = (str(row["case_id"]), str(row["slice_id"]))
            overlay_map[key] = str(row.get("overlay_path", ""))

    return {
        "model_tag": model_tag,
        "model_name": model_name,
        "final_root": final_root,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "metrics_per_class": metrics_per_class,
        "metrics_per_sample": metrics_per_sample,
        "inference_records": inference_records,
        "overlay_map": overlay_map,
        "train_log": train_log,
    }


def _bar_labels(ax, bars) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_overall_metrics(unet: Dict[str, object], trans: Dict[str, object], save_path: Path) -> None:
    rows = []
    for result in [unet, trans]:
        summary = result["summary_csv"]
        if summary is None or summary.empty:
            continue
        row = summary.iloc[0]
        rows.append((result["model_name"], float(row["mean_dice"]), float(row["mean_iou"])))

    if len(rows) < 2:
        warnings.warn("Skipping overall metrics plot because summary CSV is incomplete.")
        return

    model_names = [row[0] for row in rows]
    dice_values = [row[1] for row in rows]
    iou_values = [row[2] for row in rows]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, dice_values, width, label="Mean Dice")
    bars2 = ax.bar(x + width / 2, iou_values, width, label="Mean IoU")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Score")
    ax.set_title("BTCV 2D Multi-Organ Overall Metrics")
    ax.legend()
    _bar_labels(ax, bars1)
    _bar_labels(ax, bars2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metric(
    unet: Dict[str, object],
    trans: Dict[str, object],
    metric_name: str,
    save_path: Path,
    title: str,
) -> None:
    unet_df = unet["metrics_per_class"]
    trans_df = trans["metrics_per_class"]
    if unet_df is None or trans_df is None or unet_df.empty or trans_df.empty:
        warnings.warn(f"Skipping {metric_name} per-class plot because metrics_per_class.csv is missing.")
        return

    merged = pd.merge(
        unet_df[["class_id", "class_name", metric_name]],
        trans_df[["class_id", "class_name", metric_name]],
        on="class_id",
        how="outer",
        suffixes=("_unet", "_trans"),
    ).sort_values("class_id")

    merged["class_name"] = merged["class_name_unet"].fillna(merged["class_name_trans"]).fillna(
        merged["class_id"].astype(int).astype(str)
    )
    merged[f"{metric_name}_unet"] = pd.to_numeric(merged[f"{metric_name}_unet"], errors="coerce").fillna(0.0)
    merged[f"{metric_name}_trans"] = pd.to_numeric(merged[f"{metric_name}_trans"], errors="coerce").fillna(0.0)

    x = np.arange(len(merged))
    width = 0.38
    fig_width = max(10, len(merged) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bars1 = ax.bar(x - width / 2, merged[f"{metric_name}_unet"], width, label=unet["model_name"])
    bars2 = ax.bar(x + width / 2, merged[f"{metric_name}_trans"], width, label=trans["model_name"])
    ax.set_xticks(x)
    ax.set_xticklabels(merged["class_name"], rotation=45, ha="right")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sample_boxplot(
    unet: Dict[str, object],
    trans: Dict[str, object],
    metric_name: str,
    save_path: Path,
    title: str,
) -> None:
    unet_df = unet["metrics_per_sample"]
    trans_df = trans["metrics_per_sample"]
    if unet_df is None or trans_df is None or metric_name not in unet_df.columns or metric_name not in trans_df.columns:
        warnings.warn(f"Skipping boxplot for {metric_name} because metrics_per_sample.csv is missing.")
        return

    data = [
        pd.to_numeric(unet_df[metric_name], errors="coerce").dropna().values,
        pd.to_numeric(trans_df[metric_name], errors="coerce").dropna().values,
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=[unet["model_name"], trans["model_name"]], showfliers=False)
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(unet_log: Optional[pd.DataFrame], trans_log: Optional[pd.DataFrame], metric: str, save_path: Path) -> None:
    if unet_log is None or trans_log is None:
        warnings.warn(f"Skipping learning curve for {metric} because one or more train logs are missing.")
        return
    required_columns = {"epoch", f"train_{metric}", f"val_{metric}"}
    if not required_columns.issubset(unet_log.columns) or not required_columns.issubset(trans_log.columns):
        warnings.warn(f"Skipping learning curve for {metric} because required columns are missing.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(unet_log["epoch"], unet_log[f"train_{metric}"], label=f"U-Net train {metric}")
    ax.plot(unet_log["epoch"], unet_log[f"val_{metric}"], label=f"U-Net val {metric}")
    ax.plot(trans_log["epoch"], trans_log[f"train_{metric}"], label=f"TransUNet train {metric}")
    ax.plot(trans_log["epoch"], trans_log[f"val_{metric}"], label=f"TransUNet val {metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.title())
    ax.set_title(f"Learning Curves: {metric.title()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_merged_sample_metrics(unet: Dict[str, object], trans: Dict[str, object]) -> Optional[pd.DataFrame]:
    unet_df = unet["metrics_per_sample"]
    trans_df = trans["metrics_per_sample"]
    if unet_df is None or trans_df is None:
        warnings.warn("Cannot build merged sample metrics because metrics_per_sample.csv is missing.")
        return None

    merged = pd.merge(
        unet_df[["case_id", "slice_id", "mean_dice", "mean_iou"]],
        trans_df[["case_id", "slice_id", "mean_dice", "mean_iou"]],
        on=["case_id", "slice_id"],
        how="outer",
        suffixes=("_unet", "_transunet"),
    )
    for col in ["mean_dice_unet", "mean_iou_unet", "mean_dice_transunet", "mean_iou_transunet"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["unet_mean_dice"] = merged["mean_dice_unet"]
    merged["unet_mean_iou"] = merged["mean_iou_unet"]
    merged["transunet_mean_dice"] = merged["mean_dice_transunet"]
    merged["transunet_mean_iou"] = merged["mean_iou_transunet"]
    merged["dice_gap"] = merged["transunet_mean_dice"] - merged["unet_mean_dice"]
    merged["iou_gap"] = merged["transunet_mean_iou"] - merged["unet_mean_iou"]
    merged = merged[
        [
            "case_id",
            "slice_id",
            "unet_mean_dice",
            "unet_mean_iou",
            "transunet_mean_dice",
            "transunet_mean_iou",
            "dice_gap",
            "iou_gap",
        ]
    ]
    return merged


def select_qualitative_samples(merged_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if merged_df is None or merged_df.empty:
        return pd.DataFrame()

    df = merged_df.copy()
    df["avg_mean_dice"] = df[["unet_mean_dice", "transunet_mean_dice"]].mean(axis=1)
    df["abs_dice_gap"] = df["dice_gap"].abs()

    picks: List[pd.Series] = []

    def add_first(sorted_df: pd.DataFrame) -> None:
        for _, row in sorted_df.iterrows():
            key = (str(row["case_id"]), str(row["slice_id"]))
            if key not in {(str(r["case_id"]), str(r["slice_id"])) for r in picks}:
                picks.append(row)
                return

    add_first(df.sort_values("avg_mean_dice", ascending=False))
    add_first(df.sort_values("dice_gap", ascending=True))
    add_first(df.sort_values("dice_gap", ascending=False))
    add_first(df.sort_values("avg_mean_dice", ascending=True))

    if len(picks) < 4:
        remaining = df.sort_values("abs_dice_gap", ascending=False)
        for _, row in remaining.iterrows():
            key = (str(row["case_id"]), str(row["slice_id"]))
            if key not in {(str(r["case_id"]), str(r["slice_id"])) for r in picks}:
                picks.append(row)
            if len(picks) >= 4:
                break

    return pd.DataFrame(picks[:4]).reset_index(drop=True)


def _resolve_overlay_path(result: Dict[str, object], case_id: str, slice_id: str) -> Optional[Path]:
    overlay_map = result.get("overlay_map", {})
    direct = overlay_map.get((case_id, slice_id))
    if direct:
        path = Path(direct)
        if path.exists():
            return path

    overlays_dir = Path(result["final_root"]) / "overlays"
    patterns = [
        f"{case_id}_{slice_id}_overlay.png",
        f"*{case_id}*{slice_id}*overlay.png",
    ]
    for pattern in patterns:
        matches = list(overlays_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def build_qualitative_grid(
    selected_df: pd.DataFrame,
    unet: Dict[str, object],
    trans: Dict[str, object],
    save_path: Path,
) -> None:
    if selected_df.empty:
        warnings.warn("Skipping qualitative comparison grid because no merged samples are available.")
        return

    rows = len(selected_df)
    fig, axes = plt.subplots(rows, 2, figsize=(12, max(4, rows * 4)))
    if rows == 1:
        axes = np.array([axes])

    for row_idx, row in selected_df.iterrows():
        case_id = str(row["case_id"])
        slice_id = str(row["slice_id"])
        unet_overlay = _resolve_overlay_path(unet, case_id, slice_id)
        trans_overlay = _resolve_overlay_path(trans, case_id, slice_id)
        titles = [
            f"U-Net\nDice={row['unet_mean_dice']:.4f} IoU={row['unet_mean_iou']:.4f}",
            f"TransUNet\nDice={row['transunet_mean_dice']:.4f} IoU={row['transunet_mean_iou']:.4f}",
        ]

        for col_idx, (overlay_path, title) in enumerate(zip([unet_overlay, trans_overlay], titles)):
            axis = axes[row_idx, col_idx]
            if overlay_path is not None and overlay_path.exists():
                image = mpimg.imread(overlay_path)
                axis.imshow(image)
            else:
                axis.text(0.5, 0.5, "Overlay missing", ha="center", va="center")
            axis.set_title(f"{case_id} | {slice_id}\n{title}")
            axis.axis("off")

    fig.suptitle("Qualitative Comparison Grid", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_comparison_summary(
    unet: Dict[str, object],
    trans: Dict[str, object],
    merged_sample_metrics: Optional[pd.DataFrame],
    output_dir: Path,
) -> None:
    summary_rows: List[Dict[str, object]] = []
    per_class_comparison: Dict[str, Dict[str, object]] = {}

    for result in [unet, trans]:
        summary_csv = result["summary_csv"]
        if summary_csv is None or summary_csv.empty:
            continue
        summary_rows.append(summary_csv.iloc[0].to_dict())

    if summary_rows:
        comparison_df = pd.DataFrame(summary_rows)
        comparison_df.to_csv(output_dir / "comparison_summary.csv", index=False)
    else:
        warnings.warn("comparison_summary.csv was not generated because no summary rows were available.")

    unet_class = unet["metrics_per_class"]
    trans_class = trans["metrics_per_class"]
    if unet_class is not None and trans_class is not None:
        merged_classes = pd.merge(
            unet_class,
            trans_class,
            on="class_id",
            how="outer",
            suffixes=("_unet", "_transunet"),
        )
        for _, row in merged_classes.iterrows():
            class_id = str(int(row["class_id"])) if not pd.isna(row["class_id"]) else "unknown"
            class_name = row.get("class_name_unet")
            if pd.isna(class_name):
                class_name = row.get("class_name_transunet")
            per_class_comparison[class_id] = {
                "class_name": class_name,
                "unet_mean_dice": None if pd.isna(row.get("mean_dice_unet")) else float(row.get("mean_dice_unet")),
                "transunet_mean_dice": None
                if pd.isna(row.get("mean_dice_transunet"))
                else float(row.get("mean_dice_transunet")),
                "unet_mean_iou": None if pd.isna(row.get("mean_iou_unet")) else float(row.get("mean_iou_unet")),
                "transunet_mean_iou": None
                if pd.isna(row.get("mean_iou_transunet"))
                else float(row.get("mean_iou_transunet")),
            }

    overall = {}
    for result in [unet, trans]:
        summary = result["summary_csv"]
        if summary is not None and not summary.empty:
            row = summary.iloc[0]
            overall[result["model_name"]] = {
                "mean_dice": float(row["mean_dice"]),
                "mean_iou": float(row["mean_iou"]),
                "num_test_samples": int(row["num_test_samples"]),
            }

    best_model_by_mean_dice = None
    best_model_by_mean_iou = None
    if overall:
        best_model_by_mean_dice = max(overall.items(), key=lambda item: item[1]["mean_dice"])[0]
        best_model_by_mean_iou = max(overall.items(), key=lambda item: item[1]["mean_iou"])[0]

    comparison_json = {
        "overall_comparison": overall,
        "per_class_comparison": per_class_comparison,
        "best_model_by_mean_dice": best_model_by_mean_dice,
        "best_model_by_mean_iou": best_model_by_mean_iou,
    }
    if merged_sample_metrics is not None:
        comparison_json["merged_sample_count"] = int(len(merged_sample_metrics))

    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(comparison_json, f, indent=2, ensure_ascii=False)

    if merged_sample_metrics is not None:
        merged_sample_metrics.to_csv(output_dir / "merged_metrics_per_sample.csv", index=False)


def main() -> None:
    warnings.simplefilter("always", UserWarning)
    output_dir = ensure_dirs()

    unet = load_model_results("UNet2D_BTCV", UNET_FINAL_ROOT, UNET_TRAIN_LOG)
    trans = load_model_results("TransUNet2D_BTCV", TRANS_FINAL_ROOT, TRANS_TRAIN_LOG)

    plot_overall_metrics(unet, trans, output_dir / "overall_metrics_bar.png")
    plot_per_class_metric(
        unet,
        trans,
        metric_name="mean_dice",
        save_path=output_dir / "per_class_dice_bar.png",
        title="Per-Class Dice Comparison",
    )
    plot_per_class_metric(
        unet,
        trans,
        metric_name="mean_iou",
        save_path=output_dir / "per_class_iou_bar.png",
        title="Per-Class IoU Comparison",
    )
    plot_sample_boxplot(
        unet,
        trans,
        metric_name="mean_dice",
        save_path=output_dir / "sample_dice_boxplot.png",
        title="Sample-Level Dice Distribution",
    )
    plot_sample_boxplot(
        unet,
        trans,
        metric_name="mean_iou",
        save_path=output_dir / "sample_iou_boxplot.png",
        title="Sample-Level IoU Distribution",
    )
    plot_learning_curves(unet["train_log"], trans["train_log"], "loss", output_dir / "learning_curves_loss.png")
    plot_learning_curves(unet["train_log"], trans["train_log"], "dice", output_dir / "learning_curves_dice.png")
    plot_learning_curves(unet["train_log"], trans["train_log"], "iou", output_dir / "learning_curves_iou.png")

    merged_sample_metrics = build_merged_sample_metrics(unet, trans)
    selected_samples = select_qualitative_samples(merged_sample_metrics)
    build_qualitative_grid(selected_samples, unet, trans, output_dir / "qualitative_comparison_grid.png")

    export_comparison_summary(unet, trans, merged_sample_metrics, output_dir)

    print(f"output_dir: {output_dir}")
    if merged_sample_metrics is not None:
        print(f"merged_samples: {len(merged_sample_metrics)}")
    if not selected_samples.empty:
        print(f"qualitative_samples: {len(selected_samples)}")
    print("Final BTCV comparison export finished.")


if __name__ == "__main__":
    main()
