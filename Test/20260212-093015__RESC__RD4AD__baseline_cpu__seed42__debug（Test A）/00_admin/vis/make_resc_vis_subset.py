import argparse, os, random
from pathlib import Path
import yaml

IMG_EXT = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def list_imgs(d: Path):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT])

def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    os.symlink(str(src), str(dst))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--resc_root", required=True)
    ap.add_argument("--cfg_base", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_normal", type=int, default=40)
    ap.add_argument("--n_abnormal", type=int, default=40)
    ap.add_argument("--out_images_dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    resc = Path(args.resc_root)
    cfg_base = Path(args.cfg_base)
    out_images_dir = Path(args.out_images_dir)

    good_dir = resc / "test/good/img"
    bad_dir  = resc / "test/ungood/img"
    mask_dir = resc / "test/ungood/label"
    train_good_dir = resc / "train/good"

    for p in [good_dir, bad_dir, mask_dir, train_good_dir]:
        if not p.is_dir():
            raise SystemExit(f"missing dir: {p}")

    good = list_imgs(good_dir)
    bad  = list_imgs(bad_dir)

    # 只保留有对应 mask 的 abnormal
    bad2 = [p for p in bad if (mask_dir / p.name).is_file()]

    rng = random.Random(args.seed)
    rng.shuffle(good)
    rng.shuffle(bad2)

    sel_good = good[: args.n_normal]
    sel_bad  = bad2[: args.n_abnormal]

    if len(sel_good) < args.n_normal:
        raise SystemExit(f"not enough normal images: need {args.n_normal}, got {len(sel_good)}")
    if len(sel_bad) < args.n_abnormal:
        raise SystemExit(f"not enough abnormal images with masks: need {args.n_abnormal}, got {len(sel_bad)}")

    subset = resc.parent / f"RESC__vis_{args.tag}"

    # train: 直接链接整个 train/good，避免 train_data.setup() 报错
    (subset / "train").mkdir(parents=True, exist_ok=True)
    if not (subset / "train/good").exists():
        safe_symlink(train_good_dir, subset / "train/good")

    # test/val: 建空结构
    for p in [
        subset/"test/good/img", subset/"test/ungood/img", subset/"test/ungood/label",
        subset/"val/good/img",  subset/"val/ungood/img",  subset/"val/ungood/label",
    ]:
        p.mkdir(parents=True, exist_ok=True)

    # 链接样本
    for p in sel_good:
        safe_symlink(p, subset/"test/good/img"/p.name)
        safe_symlink(p, subset/"val/good/img"/p.name)

    for p in sel_bad:
        safe_symlink(p, subset/"test/ungood/img"/p.name)
        safe_symlink(p, subset/"val/ungood/img"/p.name)
        safe_symlink(mask_dir/p.name, subset/"test/ungood/label"/p.name)
        safe_symlink(mask_dir/p.name, subset/"val/ungood/label"/p.name)

    # patch config: root/path 绝对路径，dir 用相对路径，防止 data/RESC/data/RESC
    cfg = yaml.safe_load(cfg_base.read_text())
    ds = cfg.get("dataset", {})
    ds["path"] = str(subset)
    ds["root"] = str(subset)
    ds["normal_dir"] = "train/good"
    ds["normal_test_dir"] = "test/good/img"
    ds["abnormal_dir"] = "test/ungood/img"
    ds["mask_dir"] = "test/ungood/label"
    ds["mask"] = "test/ungood/label"
    ds["train_batch_size"] = 1
    ds["eval_batch_size"] = 1
    ds["inference_batch_size"] = 1
    ds["num_workers"] = 0
    cfg["dataset"] = ds

    # 尝试开启保存可视化
    vis = cfg.get("visualization", {})
    vis["show_images"] = False
    vis["log_images"] = False
    vis["save_images"] = True
    vis["image_save_path"] = str(out_images_dir / f"vis_{args.tag}")
    vis["num_images"] = args.n_normal + args.n_abnormal
    vis["max_images"] = args.n_normal + args.n_abnormal
    cfg["visualization"] = vis

    out_cfg = run_dir / f"06_outputs/results/reverse_distillation/RESC/run/config__vis_{args.tag}.yaml"
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 写 index 便于论文引用
    idx = out_images_dir / f"vis_{args.tag}" / "index__subset.txt"
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text(
        "".join([f"normal\t{p}\n" for p in sel_good] + [f"abnormal\t{p}\n" for p in sel_bad]),
        encoding="utf-8"
    )

    print("subset_root =", subset)
    print("out_cfg     =", out_cfg)
    print("out_images  =", vis["image_save_path"])
    print("n_normal    =", len(sel_good))
    print("n_abnormal  =", len(sel_bad))

if __name__ == "__main__":
    main()
