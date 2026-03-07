#!/usr/bin/env python3
"""Generate B0 debug configs for GBR-AD on RESC and OCT2017 stubs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def _ensure_anomalib_on_path(bmad_dir: Path) -> None:
    src = bmad_dir / "anomalib" / "src"
    if not src.is_dir():
        raise SystemExit(f"anomalib src path not found: {src}")
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _prepare_model_config_lookup(run_dir: Path, bmad_dir: Path) -> None:
    src_cfg = bmad_dir / "anomalib" / "src" / "anomalib" / "models" / "gbr_ad" / "config.yaml"
    if not src_cfg.is_file():
        raise SystemExit(f"gbr_ad config not found: {src_cfg}")

    dst_cfg = run_dir / "src" / "anomalib" / "models" / "gbr_ad" / "config.yaml"
    dst_cfg.parent.mkdir(parents=True, exist_ok=True)

    if dst_cfg.exists() or dst_cfg.is_symlink():
        return
    dst_cfg.symlink_to(src_cfg)


def _set_common_overrides(cfg, run_dir: Path, dataset_name: str, stub_root: Path) -> None:
    cfg.model.name = "gbr_ad"
    cfg.project.unique_dir = True
    cfg.project.path = str(run_dir / "06_outputs" / "results" / "gbr_ad" / dataset_name / "run")

    cfg.trainer.accelerator = "cpu"
    cfg.trainer.devices = 1
    cfg.trainer.max_epochs = 1
    cfg.trainer.limit_train_batches = 5
    cfg.trainer.limit_val_batches = 5
    cfg.trainer.limit_test_batches = 5

    cfg.dataset.format = "folder"
    cfg.dataset.name = dataset_name
    cfg.dataset.root = str(stub_root)
    cfg.dataset.path = str(stub_root)
    cfg.dataset.normal_dir = "train/good/data"
    cfg.dataset.test_dir = "test/good/data"
    cfg.dataset.normal_test_dir = "test/good/data"
    cfg.dataset.abnormal_dir = "test/ungood/data"


def _generate_one(dataset_name: str, run_dir: Path, out_path: Path):
    from anomalib.config import get_configurable_parameters

    stub_root = run_dir / "03_data" / "__b0_stub__" / dataset_name
    cfg = get_configurable_parameters(model_name="gbr_ad")
    _set_common_overrides(cfg, run_dir, dataset_name, stub_root)

    if dataset_name == "RESC":
        cfg.dataset.task = "segmentation"
        cfg.dataset.mask_dir = "test/ungood/label"
    elif dataset_name == "OCT2017":
        cfg.dataset.task = "classification"
        if "mask_dir" in cfg.dataset:
            del cfg.dataset["mask_dir"]
    else:
        raise SystemExit(f"Unsupported dataset: {dataset_name}")

    OmegaConf.save(cfg, out_path)


def _validate_yaml(path: Path, run_dir: Path, dataset_name: str) -> None:
    cfg = OmegaConf.load(path)
    expected_project_path = str(run_dir / "06_outputs" / "results" / "gbr_ad" / dataset_name / "run")
    print(f"[CFG] {path.name}")
    print(f"  exists={path.is_file()}")
    print(f"  model.name={cfg.model.name}")
    print(f"  project.unique_dir={cfg.project.unique_dir}")
    print(f"  project.path={cfg.project.path}")
    print(
        "  trainer="
        f"accelerator={cfg.trainer.accelerator},devices={cfg.trainer.devices},max_epochs={cfg.trainer.max_epochs},"
        f"limit_train_batches={cfg.trainer.limit_train_batches},"
        f"limit_val_batches={cfg.trainer.limit_val_batches},"
        f"limit_test_batches={cfg.trainer.limit_test_batches}"
    )
    print(
        "  dataset="
        f"name={cfg.dataset.name},task={cfg.dataset.task},path={cfg.dataset.path},root={cfg.dataset.root},"
        f"normal_dir={cfg.dataset.normal_dir},test_dir={cfg.dataset.test_dir},"
        f"abnormal_dir={cfg.dataset.abnormal_dir}"
    )
    has_mask = "mask_dir" in cfg.dataset
    print(f"  mask_dir_present={has_mask}")
    if has_mask:
        print(f"  mask_dir={cfg.dataset.mask_dir}")

    checks = [
        ("model.name == gbr_ad", cfg.model.name == "gbr_ad"),
        ("project.unique_dir == True", bool(cfg.project.unique_dir) is True),
        ("project.path exact", cfg.project.path == expected_project_path),
        ("trainer.accelerator == cpu", str(cfg.trainer.accelerator) == "cpu"),
        ("trainer.devices == 1", int(cfg.trainer.devices) == 1),
        ("trainer.max_epochs == 1", int(cfg.trainer.max_epochs) == 1),
        ("trainer.limit_train_batches == 5", int(cfg.trainer.limit_train_batches) == 5),
        ("trainer.limit_val_batches == 5", int(cfg.trainer.limit_val_batches) == 5),
        ("trainer.limit_test_batches == 5", int(cfg.trainer.limit_test_batches) == 5),
        ("dataset.test_dir set", str(cfg.dataset.get("test_dir", "")) == "test/good/data"),
    ]
    if dataset_name == "RESC":
        checks.extend(
            [
                ("dataset.task == segmentation", str(cfg.dataset.task) == "segmentation"),
                ("mask_dir present", has_mask),
            ]
        )
    else:
        checks.extend(
            [
                ("dataset.task == classification", str(cfg.dataset.task) == "classification"),
                ("mask_dir absent", not has_mask),
            ]
        )
    for label, ok in checks:
        print(f"  check: {label} => {'OK' if ok else 'FAIL'}")


def main() -> int:
    run_dir = Path(_require_env("RUN_DIR")).resolve()
    bmad_dir = Path(_require_env("BMAD_DIR")).resolve()
    _ensure_anomalib_on_path(bmad_dir)
    _prepare_model_config_lookup(run_dir, bmad_dir)
    os.chdir(run_dir)

    config_dir = run_dir / "04_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    resc_cfg = config_dir / "RESC__GBR-AD__B0_debug.yaml"
    oct_cfg = config_dir / "OCT2017__GBR-AD__B0_debug.yaml"

    _generate_one("RESC", run_dir, resc_cfg)
    _generate_one("OCT2017", run_dir, oct_cfg)

    _validate_yaml(resc_cfg, run_dir, "RESC")
    _validate_yaml(oct_cfg, run_dir, "OCT2017")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
