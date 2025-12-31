import json
import os
import glob
import shutil
import numpy as np


SRC_DIR = "data/pickplace_scripted_npz"
DST_DIR = "data/pickplace_scripted_npz_filtered"

# 过滤规则（建议先用这套，最稳）
MIN_LEN = 10           # 太短的几乎没信息量
DROP_TOP_FRAC = 0.10   # 丢掉最长的 10%（长尾通常是“磨时间/卡阶段”）


def _ensure_empty_dir(path: str):
    os.makedirs(path, exist_ok=True)
    # 清空旧内容（避免混数据）
    for fn in os.listdir(path):
        fp = os.path.join(path, fn)
        if os.path.isfile(fp):
            os.remove(fp)


def _load_len(npz_path: str) -> int:
    with np.load(npz_path, allow_pickle=True) as f:
        # 兼容：有 episode_len 就用，没有就用 action 长度
        if "episode_len" in f.files:
            return int(f["episode_len"])
        if "action" in f.files:
            return int(f["action"].shape[0])
        if "act" in f.files:
            return int(f["act"].shape[0])
        raise KeyError(f"Cannot infer episode length from {npz_path}, keys={f.files}")


def _percentile(x, p):
    return float(np.percentile(np.asarray(x, dtype=np.float32), p))


def main():
    src_paths = sorted(glob.glob(os.path.join(SRC_DIR, "episode_*.npz")))
    if not src_paths:
        raise FileNotFoundError(f"No episode_*.npz found in: {SRC_DIR}")

    lengths = []
    for p in src_paths:
        lengths.append(_load_len(p))

    lengths_np = np.asarray(lengths, dtype=np.int32)
    n = len(lengths_np)

    p10 = _percentile(lengths_np, 10)
    p25 = _percentile(lengths_np, 25)
    p50 = _percentile(lengths_np, 50)
    p75 = _percentile(lengths_np, 75)
    p90 = _percentile(lengths_np, 90)
    p95 = _percentile(lengths_np, 95)

    # 分位数阈值：丢掉最长 DROP_TOP_FRAC
    keep_max = int(np.ceil(_percentile(lengths_np, 100 * (1 - DROP_TOP_FRAC))))

    print("==== Dataset length stats (before) ====")
    print(f"Dir: {SRC_DIR}")
    print(f"N episodes: {n}")
    print(f"min={int(lengths_np.min())}, max={int(lengths_np.max())}, mean={float(lengths_np.mean()):.1f}")
    print(f"p10={p10:.1f}, p25={p25:.1f}, p50={p50:.1f}, p75={p75:.1f}, p90={p90:.1f}, p95={p95:.1f}")
    print("")
    print("==== Filter rule ====")
    print(f"Keep if: len >= {MIN_LEN} AND len <= {keep_max}  (drop top {int(DROP_TOP_FRAC*100)}%)")
    print("")

    keep = []
    drop = []
    for p, L in zip(src_paths, lengths_np.tolist()):
        if L < MIN_LEN or L > keep_max:
            drop.append((p, L))
        else:
            keep.append((p, L))

    print("==== Filter result ====")
    print(f"Kept: {len(keep)}/{n} ({len(keep)/n:.2%})")
    print(f"Dropped: {len(drop)}/{n} ({len(drop)/n:.2%})")
    if drop:
        # 展示一些被丢的长度（不刷屏）
        drop_lens = [L for _, L in drop]
        print(f"Dropped len range: min={min(drop_lens)}, max={max(drop_lens)}")
        # 打印最极端的 10 个
        drop_sorted = sorted(drop, key=lambda x: x[1])
        print("Examples dropped (shortest 5):", [x[1] for x in drop_sorted[:5]])
        print("Examples dropped (longest 5):", [x[1] for x in drop_sorted[-5:]])
    print("")

    # 准备新目录（清空）
    _ensure_empty_dir(DST_DIR)

    # 拷贝并重新编号
    new_manifest = {
        "source_dir": SRC_DIR,
        "filter": {
            "min_len": MIN_LEN,
            "drop_top_frac": DROP_TOP_FRAC,
            "keep_max_len": keep_max,
        },
        "episodes": [],
    }

    for i, (src, L) in enumerate(keep):
        dst_name = f"episode_{i:06d}.npz"
        dst_path = os.path.join(DST_DIR, dst_name)
        shutil.copy2(src, dst_path)
        new_manifest["episodes"].append(
            {
                "file": dst_name,
                "src_file": os.path.basename(src),
                "episode_len": int(L),
            }
        )

    manifest_path = os.path.join(DST_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(new_manifest, f, indent=2, ensure_ascii=False)

    # after stats
    kept_lens = np.asarray([L for _, L in keep], dtype=np.int32)
    print("==== Dataset length stats (after) ====")
    print(f"Dir: {DST_DIR}")
    print(f"N episodes: {len(keep)}")
    print(f"min={int(kept_lens.min())}, max={int(kept_lens.max())}, mean={float(kept_lens.mean()):.1f}")
    print(f"p50={_percentile(kept_lens, 50):.1f}, p90={_percentile(kept_lens, 90):.1f}, p95={_percentile(kept_lens, 95):.1f}")
    print("")
    print(f"✅ Wrote filtered dataset to: {DST_DIR}")
    print(f"✅ Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
