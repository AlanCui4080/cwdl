
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from genmorse import compute_morse
from spectrogram import (compute_spectrogram, slice_blocks,
                         TIME_PIXEL_UNIT)

NOISE_DB = (18, 15, 12, 9, 6, 3, 0, -3, -6, -9, -12, -15)

TRAIN_WPMS = (30, 40, 50, 60)
EVAL_WPMS = (10, 25, 30, 35, 40, 45, 50, 55, 60, 65, 80)

TRAIN_PER_CAT = 3000
EVAL_PER_CAT = 300

POOL_SPLIT = (0.6, 0.2, 0.2)

DATASET_DIR = Path(__file__).parent / "dataset"
RANDOM50_TXT = DATASET_DIR / "random50" / "random50.txt"
REALCOMM_TXT = DATASET_DIR / "realcomm" / "realcomm.txt"

ROOT = Path(__file__).parent
CENTER_FREQ = 1000.0

BLOCK_W = TIME_PIXEL_UNIT * 2
BLOCK_HOP = TIME_PIXEL_UNIT

INVALID_LABEL = "-"

# 每 sequence 一行:blocks 存为单个 .npy (uint8 [K,16,128]),无 PNG 中转
CSV_COLUMNS = (
    "path", "text", "wpm", "noise_db", "source",
    "num_blocks", "height", "width",
)

def _load_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _load_tagged(path: Path, source: str) -> list[tuple[str, str]]:
    return [(ln, source) for ln in _load_lines(path)]

def _safe_filename(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:32]

def _category_dir(root: Path, wpm: int, noise_db: int) -> Path:
    noise_tag = f"noise{noise_db:+03d}db".replace("+", "p").replace("-", "m")
    return root / f"wpm{wpm}" / noise_tag

def _split_pool(pool: list[tuple[str, str]],
                ratios: tuple[float, float, float]
                ) -> tuple[list[tuple[str, str]], ...]:
    n = len(pool)
    perm = np.random.permutation(n)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train = [pool[i] for i in perm[:n_train]]
    val = [pool[i] for i in perm[n_train:n_train + n_val]]
    test = [pool[i] for i in perm[n_train + n_val:]]
    return train, val, test

def _sample_no_replace(pool: list[tuple[str, str]], n: int
                       ) -> list[tuple[str, str]]:
    if n > len(pool):
        raise ValueError(f"pool ({len(pool)}) < requested ({n})")
    idx = np.random.permutation(len(pool))[:n]
    return [pool[i] for i in idx]

def _worker_init():

    seed = (int.from_bytes(os.urandom(4), "little") ^ (os.getpid() << 16)) % (2**32)
    np.random.seed(seed)

def _gen_sequence(task: tuple) -> tuple:
    text, source, j, cdir, out_root, wpm, noise_db = task
    audio = compute_morse(text, wpm=float(wpm), noise_db=float(noise_db))
    img = compute_spectrogram(audio, center_freq=CENTER_FREQ)
    blocks = slice_blocks(img, BLOCK_W, BLOCK_HOP)

    safe = _safe_filename(text)
    h = img.shape[0]
    # [K,16,128] uint8,无损,单文件;训练端 np.load 直接得张量
    stacked = np.stack([blk for _, _, blk in blocks], axis=0)
    fname = f"{j:06d}_{safe}.npy"
    fpath = cdir / fname
    np.save(fpath, stacked)
    rel = fpath.relative_to(out_root).as_posix()
    row = (rel, text, wpm, noise_db, source, len(blocks), h, BLOCK_W)
    return row

def _build_split(out_root: Path, wpms, per_cat: int,
                 pool: list[tuple[str, str]],
                 workers: int = 0) -> None:
    if per_cat > len(pool):
        raise RuntimeError(
            f"pool {len(pool)} < per_cat {per_cat}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "index.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(CSV_COLUMNS)

    n_cats = len(wpms) * len(NOISE_DB)
    total_seqs = n_cats * per_cat
    print(f"  -> {out_root.name}: {n_cats} cats x {per_cat} seqs = "
          f"{total_seqs}  (random50 + realcomm)  "
          f"block={BLOCK_W}px hop={BLOCK_HOP}px  csv={csv_path.name}")

    nw = workers if workers > 0 else os.cpu_count() or 4
    cat_idx = 0
    seqs_done = 0
    blocks_done = 0
    try:
        with ProcessPoolExecutor(max_workers=nw, initializer=_worker_init) as ex:
            for wpm in wpms:
                for noise_db in NOISE_DB:
                    cat_idx += 1
                    cdir = _category_dir(out_root, wpm, noise_db)
                    cdir.mkdir(parents=True, exist_ok=True)

                    items = _sample_no_replace(pool, per_cat)
                    tasks = [(text, source, j, cdir, out_root, wpm, noise_db)
                             for j, (text, source) in enumerate(items)]

                    cat_blocks = 0
                    futures = [ex.submit(_gen_sequence, t) for t in tasks]
                    for fut in as_completed(futures):
                        row = fut.result()
                        writer.writerow(row)
                        cat_blocks += row[5]
                        blocks_done += row[5]
                        seqs_done += 1
                        if seqs_done % 1000 == 0:
                            pct = 100.0 * seqs_done / total_seqs
                            print(f"     [{cat_idx}/{n_cats}] wpm={wpm} "
                                  f"noise={noise_db:+d}dB  seqs={seqs_done}/"
                                  f"{total_seqs} ({pct:.1f}%)  "
                                  f"blocks={blocks_done}")
                    csv_file.flush()
                    print(f"     [{cat_idx}/{n_cats}] wpm={wpm} "
                          f"noise={noise_db:+d}dB  -> "
                          f"{cdir.relative_to(out_root.parent)}  "
                          f"({per_cat} seqs, {cat_blocks} blocks)")
    finally:
        csv_file.close()

    print(f"  {out_root.name}: done {seqs_done} seqs / {blocks_done} blocks, "
          f"csv -> {csv_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0,
                    help="并行进程数 (0=自动, 默认 CPU 核数)")
    args = ap.parse_args()

    pool_random = _load_tagged(RANDOM50_TXT, "random50")
    pool_real = _load_tagged(REALCOMM_TXT, "realcomm")
    print(f"random50 pool: {len(pool_random)}")
    print(f"realcomm pool: {len(pool_real)}")

    # 两个 pool 分别 split 后合并, 保证 realcomm 在 train/val/test 中均有分布
    r_train, r_val, r_test = _split_pool(pool_random, POOL_SPLIT)
    c_train, c_val, c_test = _split_pool(pool_real, POOL_SPLIT)

    p_train = r_train + c_train
    p_val = r_val + c_val
    p_test = r_test + c_test

    print(f"combined split: train={len(p_train)} val={len(p_val)} "
          f"test={len(p_test)}")

    print(f"workers: {args.workers if args.workers>0 else os.cpu_count()}")
    print(f"NOISE_DB: {NOISE_DB}")
    print(f"TRAIN_WPMS: {TRAIN_WPMS}")
    print(f"EVAL_WPMS: {EVAL_WPMS}")

    print("\n[cnntriset/trainset]")
    _build_split(ROOT / "cnntriset" / "trainset", TRAIN_WPMS, TRAIN_PER_CAT,
                 p_train, workers=args.workers)

    print("\n[cnntriset/valset]")
    _build_split(ROOT / "cnntriset" / "valset", EVAL_WPMS, EVAL_PER_CAT,
                 p_val, workers=args.workers)

    print("\n[cnntriset/testset]")
    _build_split(ROOT / "cnntriset" / "testset", EVAL_WPMS, EVAL_PER_CAT,
                 p_test, workers=args.workers)

    print("\nall sets generated.")

if __name__ == "__main__":
    main()
