import argparse, os, re, json
from pathlib import Path
import numpy as np

IMG_EXT = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
MAP_EXT = {".npy",".npz",".png"}

def list_images(d: Path):
    if not d.exists(): return []
    out = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            # 排除 mask 目录里的 png
            if any(k in str(p).lower() for k in ["anomaly_mask","ground_truth","groundtruth","mask","masks","gt","label","labels"]):
                continue
            out.append(p)
    return sorted(out)

def find_mask_dir(ungood_dir: Path):
    # 先尝试常见目录名
    cand = [
        ungood_dir/"anomaly_mask",
        ungood_dir/"ground_truth",
        ungood_dir/"groundtruth",
        ungood_dir/"mask",
        ungood_dir/"masks",
        ungood_dir/"gt",
        ungood_dir/"labels",
        ungood_dir/"label",
    ]
    for c in cand:
        if c.exists():
            return c
    # 再按目录内文件数搜索
    best = None
    best_n = 0
    for d in ungood_dir.rglob("*"):
        if d.is_dir():
            name = d.name.lower()
            if any(k in name for k in ["mask","gt","ground","label","anomaly"]):
                n = sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXT)
                if n > best_n:
                    best, best_n = d, n
    return best

def imread_gray(p: Path):
    from PIL import Image
    a = np.array(Image.open(p).convert("L"), dtype=np.float32)
    return a

def load_map(p: Path):
    suf = p.suffix.lower()
    if suf == ".npy":
        a = np.load(p)
        return a.squeeze().astype(np.float32)
    if suf == ".npz":
        z = np.load(p)
        k = list(z.keys())[0]
        a = z[k]
        return a.squeeze().astype(np.float32)
    if suf == ".png":
        return imread_gray(p)
    raise ValueError(f"Unsupported map: {p}")

def resize_to(a: np.ndarray, shape_hw):
    if a.shape == shape_hw:
        return a
    from PIL import Image
    img = Image.fromarray(a.astype(np.float32))
    img = img.resize((shape_hw[1], shape_hw[0]), resample=Image.BILINEAR)
    return np.array(img, dtype=np.float32)

def binarize_mask(a: np.ndarray):
    # mask 只要 >0 视为异常
    return (a > 0).astype(np.uint8)

def norm_key(stem: str):
    s = stem.lower()
    # 去掉常见后缀
    for suf in ["_anomaly_map","_anomalymap","_heatmap","_map","_pred","_score","_amap","_mask"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

def index_files(root: Path, exts, prefer_keywords=None):
    idx = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            k = norm_key(p.stem)
            # 同 key 多个文件时，选路径更短且更含关键词的
            if k not in idx:
                idx[k] = p
            else:
                old = idx[k]
                def score(x):
                    s = 0
                    sp = str(x).lower()
                    if prefer_keywords:
                        for w in prefer_keywords:
                            if w in sp: s += 5
                    s -= len(sp)
                    return s
                if score(p) > score(old):
                    idx[k] = p
    return idx

def auc_rank(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    pos = scores[labels==1]
    neg = scores[labels==0]
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(n_pos, np.uint8), np.zeros(n_neg, np.uint8)])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size+1, dtype=np.float64)
    s_sorted = all_scores[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while j+1 < s_sorted.size and s_sorted[j+1] == s_sorted[i]:
            j += 1
        if j > i:
            r = (i+1 + j+1)/2.0
            ranks[order[i:j+1]] = r
        i = j+1
    sum_r_pos = ranks[all_labels==1].sum()
    auc = (sum_r_pos - n_pos*(n_pos+1)/2.0)/(n_pos*n_neg)
    return float(auc)

def try_cc(mask: np.ndarray):
    # 返回每个连通区域的 bool mask 列表
    m = mask.astype(np.uint8)
    try:
        import cv2
        num, lab = cv2.connectedComponents(m, connectivity=8)
        return [(lab == i) for i in range(1, num)]
    except Exception:
        pass
    try:
        from skimage.measure import label
        lab = label(m, connectivity=2)
        mx = int(lab.max())
        return [(lab == i) for i in range(1, mx+1)]
    except Exception:
        pass
    try:
        from scipy.ndimage import label as ndlabel
        lab, mx = ndlabel(m)
        return [(lab == i) for i in range(1, int(mx)+1)]
    except Exception:
        pass
    # 兜底 BFS，慢但可用
    h,w = m.shape
    vis = np.zeros((h,w), bool)
    comps = []
    for y in range(h):
        for x in range(w):
            if m[y,x] and not vis[y,x]:
                q=[(y,x)]
                vis[y,x]=True
                pts=[]
                while q:
                    cy,cx=q.pop()
                    pts.append((cy,cx))
                    for ny in (cy-1,cy,cy+1):
                        for nx in (cx-1,cx,cx+1):
                            if 0<=ny<h and 0<=nx<w and (ny!=cy or nx!=cx):
                                if m[ny,nx] and not vis[ny,nx]:
                                    vis[ny,nx]=True
                                    q.append((ny,nx))
                cm=np.zeros((h,w), bool)
                yy,xx=zip(*pts)
                cm[list(yy), list(xx)]=True
                comps.append(cm)
    return comps

def choose_pred_dir(run_dir: Path, target_min):
    # 搜索 run_dir 下可能的热图目录，按 “map 文件数量接近 target_min” 排序
    cands = []
    for d in run_dir.rglob("*"):
        if d.is_dir():
            name = str(d).lower()
            if not any(k in name for k in ["map","heat","anomaly","vis","result","output","pred"]):
                continue
            n = sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in MAP_EXT)
            if n >= max(10, int(0.3*target_min)):
                cands.append((abs(n-target_min), -n, str(d), n))
    cands.sort()
    return cands[:20]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--resc_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred_dir", default="")
    ap.add_argument("--max_fpr", type=float, default=0.3)
    ap.add_argument("--n_thr", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)  # 像素下采样，1 为全量
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    resc_dir = Path(args.resc_dir)
    split = args.split

    good_dir = resc_dir/split/"good"
    ungood_dir = resc_dir/split/"ungood"

    good_imgs = list_images(good_dir)
    ungood_imgs = list_images(ungood_dir)

    if len(good_imgs)==0 and len(ungood_imgs)==0:
        raise SystemExit(f"没有找到 {resc_dir}/{split} 的图像文件")

    mask_dir = find_mask_dir(ungood_dir)
    if mask_dir is None or (not mask_dir.exists()):
        raise SystemExit("未找到 GT mask 目录，Pixel AUROC 与 PRO 无法计算。请确认 test/ungood 下是否有 anomaly_mask 或类似目录。")

    mask_idx = index_files(mask_dir, IMG_EXT, prefer_keywords=["mask","gt","anomaly"])
    n_mask = len(mask_idx)

    total_imgs = len(good_imgs) + len(ungood_imgs)
    target = max(len(ungood_imgs), total_imgs)

    if args.pred_dir:
        pred_dir = Path(args.pred_dir)
        if not pred_dir.exists():
            raise SystemExit(f"pred_dir 不存在: {pred_dir}")
    else:
        cands = choose_pred_dir(run_dir, target)
        if not cands:
            raise SystemExit("未自动找到预测热图目录。你需要手动提供 --pred_dir 指向保存热图的目录。")
        pred_dir = Path(cands[0][2])

    pred_idx = index_files(pred_dir, MAP_EXT, prefer_keywords=["map","heat","anomaly"])

    def get_pred_for(img_path: Path):
        k = norm_key(img_path.stem)
        if k in pred_idx:
            return pred_idx[k]
        # 兜底：尝试前缀匹配
        for kk,pp in pred_idx.items():
            if kk.startswith(k) or k.startswith(kk):
                return pp
        return None

    def get_mask_for(img_path: Path):
        k = norm_key(img_path.stem)
        if k in mask_idx:
            return mask_idx[k]
        for kk,pp in mask_idx.items():
            if kk.startswith(k) or k.startswith(kk):
                return pp
        return None

    # 收集 image-level
    img_scores = []
    img_labels = []

    # 收集 pixel-level
    px_scores = []
    px_labels = []

    stride = max(1, int(args.stride))

    missing_pred = 0
    missing_mask = 0
    total_regions = 0

    # 先用一张 mask 决定目标大小
    sample_mask = next(iter(mask_idx.values()))
    gt_shape = binarize_mask(imread_gray(sample_mask)).shape

    # 预先加载所有预测 map 跟 gt mask，供 PRO 用
    maps_for_pro = []
    gts_for_pro = []

    def add_one(img_path: Path, label: int):
        nonlocal missing_pred, missing_mask
        pred_p = get_pred_for(img_path)
        if pred_p is None:
            missing_pred += 1
            return
        amap = load_map(pred_p)
        amap = resize_to(amap, gt_shape)

        if label == 1:
            mp = get_mask_for(img_path)
            if mp is None:
                missing_mask += 1
                return
            gt = binarize_mask(imread_gray(mp))
        else:
            gt = np.zeros(gt_shape, np.uint8)

        # image score 用 max，更接近常用 anomaly score 定义
        img_scores.append(float(np.max(amap)))
        img_labels.append(int(label))

        # pixel 用 stride 下采样减轻内存
        a_sub = amap[::stride, ::stride].ravel()
        g_sub = gt[::stride, ::stride].ravel()
        px_scores.append(a_sub.astype(np.float32))
        px_labels.append(g_sub.astype(np.uint8))

        maps_for_pro.append(amap.astype(np.float32))
        gts_for_pro.append(gt.astype(np.uint8))

    for p in good_imgs:
        add_one(p, 0)
    for p in ungood_imgs:
        add_one(p, 1)

    if missing_pred > 0:
        print(f"[WARN] 缺少预测热图: {missing_pred} 张。pred_dir 可能选错。")
        cands = choose_pred_dir(run_dir, target)
        print("候选 pred_dir (越靠前越可能正确):")
        for _,__,d,n in cands[:10]:
            print(f"  {n:6d}  {d}")
        raise SystemExit("请用 --pred_dir 手动指定正确的热图目录后重跑。")

    if missing_mask > 0:
        raise SystemExit(f"缺少 GT mask: {missing_mask} 张。mask 文件名可能不匹配对应图像。")

    # Image AUROC
    image_auroc = auc_rank(np.array(img_scores), np.array(img_labels))

    # Pixel AUROC
    px_scores_all = np.concatenate(px_scores, axis=0)
    px_labels_all = np.concatenate(px_labels, axis=0)
    pixel_auroc = auc_rank(px_scores_all, px_labels_all)

    # PRO(AUPRO)
    # 阈值选分位数，避免极值导致曲线退化
    all_vals = np.concatenate([m.ravel()[::max(1, (m.size//50000))] for m in maps_for_pro]).astype(np.float32)
    qs = np.linspace(0.0, 1.0, args.n_thr)
    thresholds = np.quantile(all_vals, qs)

    neg_total = 0
    for gt in gts_for_pro:
        neg_total += int((gt == 0).sum())

    fprs = []
    pros = []
    for thr in thresholds:
        fp = 0
        pro_sum = 0.0
        pro_cnt = 0
        for amap, gt in zip(maps_for_pro, gts_for_pro):
            pred = (amap >= thr)
            fp += int(((gt == 0) & pred).sum())
            if gt.max() == 0:
                continue
            comps = try_cc(gt)
            for cm in comps:
                area = int(cm.sum())
                if area == 0:
                    continue
                inter = int((pred & cm).sum())
                pro_sum += inter / area
                pro_cnt += 1
        fpr = fp / max(1, neg_total)
        pro = pro_sum / max(1, pro_cnt)
        fprs.append(float(fpr))
        pros.append(float(pro))

    # 只积分 FPR<=max_fpr，按 FPR 排序后梯形积分，归一化
    max_fpr = float(args.max_fpr)
    pts = sorted([(f,p) for f,p in zip(fprs, pros) if f <= max_fpr], key=lambda x: x[0])
    if len(pts) < 2:
        aupro = float("nan")
    else:
        xs = np.array([x for x,_ in pts], np.float64)
        ys = np.array([y for _,y in pts], np.float64)
        aupro = float(np.trapz(ys, xs) / max_fpr)

    out = {
        "run": run_dir.name,
        "dataset": "RESC",
        "split": split,
        "pred_dir": str(pred_dir),
        "mask_dir": str(mask_dir),
        "image_AUROC": float(image_auroc),
        "pixel_AUROC": float(pixel_auroc),
        "AUPRO": float(aupro),
        "pixel_stride": int(stride),
        "n_images_total": int(total_imgs),
        "n_images_good": int(len(good_imgs)),
        "n_images_ungood": int(len(ungood_imgs)),
        "n_masks": int(n_mask),
    }

    out_path = run_dir/"00_admin/metrics/metrics__RESC__RD4AD__test.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Image-level AUROC =", out["image_AUROC"])
    print("Pixel-level AUROC =", out["pixel_AUROC"])
    print("PRO(AUPRO)        =", out["AUPRO"])
    print("Saved to          =", str(out_path))
    print(f"Latex row: baseline_debug & {out['image_AUROC']:.4f} & {out['pixel_AUROC']:.4f} & {out['AUPRO']:.4f} \\\\")
if __name__ == "__main__":
    main()
