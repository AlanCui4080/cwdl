
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

from genmorse import compute_morse, SAMPLE_RATE
from spectrogram import (compute_spectrogram, slice_blocks,
                         MS_PER_PIXEL, TIME_PIXEL_UNIT)

NOISE_DB = (18, 15, 12, 9, 6, 3, 0, -3, -6, -9)

TRAIN_WPMS = (30, 40, 50, 60)
EVAL_WPMS = (25, 30, 35, 40, 45, 50, 55, 60, 65)

TRAIN_PER_CAT = 3000
EVAL_PER_CAT = 300
WORD_NUM_RATIO = (1, 5)

POOL_SPLIT = (0.6, 0.2, 0.2)

DATASET_DIR = Path(__file__).parent / "dataset"
OXFORD_TXT = DATASET_DIR / "oxford5000" / "oxford5000.txt"
RADIOABBR_TXT = DATASET_DIR / "radioabbr" / "radioabbr.txt"
RANDOM6_TXT = DATASET_DIR / "random6" / "random6.txt"

ROOT = Path(__file__).parent
CENTER_FREQ = 1000.0

BLOCK_W = TIME_PIXEL_UNIT * 2
BLOCK_HOP = TIME_PIXEL_UNIT

INVALID_LABEL = "-"

CSV_COLUMNS = (
    "path", "text", "label", "wpm", "noise_db", "source",
    "height", "width", "block", "block_start_px", "block_end_px",
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

def _block_label(text: str,
                 char_spans_px: list[tuple[str, int, int]],
                 start_px: int, end_px: int) -> str:

    if not char_spans_px:
        return "-" * len(text)
    chars = []
    for ch, s, e in char_spans_px:
        if s >= start_px and e <= end_px:
            chars.append(ch)
        else:
            chars.append("-")
    return "".join(chars)

def _spans_samples_to_px(char_spans: list[tuple[str, int, int]]
                         ) -> list[tuple[str, int, int]]:
    hop_samples = int(round(SAMPLE_RATE * MS_PER_PIXEL / 1000.0))
    out = []
    for ch, s, e in char_spans:
        s_px = s // hop_samples
        e_px = (e + hop_samples - 1) // hop_samples
        out.append((ch, s_px, e_px))
    return out

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

def _gen_sequence(task: tuple) -> list[tuple]:

    text, source, j, cdir, out_root, wpm, noise_db = task
    audio, char_spans = compute_morse(
        text, wpm=float(wpm), noise_db=float(noise_db),
        return_spans=True,
    )
    img = compute_spectrogram(audio, center_freq=CENTER_FREQ)
    spans_px = _spans_samples_to_px(char_spans)
    blocks = slice_blocks(img, BLOCK_W, BLOCK_HOP)

    safe = _safe_filename(text)
    h = img.shape[0]
    rows = []
    for bi, (s_px, e_px, blk) in enumerate(blocks):
        label = _block_label(text, spans_px, s_px, e_px)
        fname = f"{j:06d}_{safe}_b{bi:02d}.png"
        fpath = cdir / fname
        Image.fromarray(blk, mode="L").save(fpath)
        rel = fpath.relative_to(out_root).as_posix()
        rows.append((
            rel, text, label, wpm, noise_db, source,
            h, BLOCK_W, bi, s_px, e_px,
        ))
    return rows

def _build_split(out_root: Path, wpms, per_cat: int,
                 word_pool: list[tuple[str, str]],
                 num_pool: list[tuple[str, str]],
                 workers: int = 0) -> None:
    n_word = max(1, int(round(per_cat * WORD_NUM_RATIO[0] / sum(WORD_NUM_RATIO))))
    n_num = per_cat - n_word

    if n_word > len(word_pool):
        raise RuntimeError(
            f"word pool {len(word_pool)} < per_cat words {n_word}"
        )
    if n_num > len(num_pool):
        raise RuntimeError(
            f"num pool {len(num_pool)} < per_cat nums {n_num}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "index.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(CSV_COLUMNS)

    n_cats = len(wpms) * len(NOISE_DB)
    total_seqs = n_cats * per_cat
    print(f"  -> {out_root.name}: {n_cats} cats x {per_cat} seqs = "
          f"{total_seqs}  (word/num per cat: {n_word}/{n_num})  "
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

                    items = (_sample_no_replace(word_pool, n_word)
                             + _sample_no_replace(num_pool, n_num))
                    tasks = [(text, source, j, cdir, out_root, wpm, noise_db)
                             for j, (text, source) in enumerate(items)]

                    cat_blocks = 0
                    futures = [ex.submit(_gen_sequence, t) for t in tasks]
                    for fut in as_completed(futures):
                        rows = fut.result()
                        for row in rows:
                            writer.writerow(row)
                            cat_blocks += 1
                            blocks_done += 1
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

    word_pool = (_load_tagged(OXFORD_TXT, "oxford5000")
                 + _load_tagged(RADIOABBR_TXT, "radioabbr"))
    num_pool = _load_tagged(RANDOM6_TXT, "random6")
    print(f"word pool: {len(word_pool)} (oxford+radioabbr)")
    print(f"num pool : {len(num_pool)} (randomnum)")

    w_train, w_val, w_test = _split_pool(word_pool, POOL_SPLIT)
    n_train, n_val, n_test = _split_pool(num_pool, POOL_SPLIT)
    print(f"word split: train={len(w_train)} val={len(w_val)} "
          f"test={len(w_test)}")
    print(f"num  split: train={len(n_train)} val={len(n_val)} "
          f"test={len(n_test)}")

    print(f"workers: {args.workers if args.workers>0 else os.cpu_count()}")

    print("\n[cnntriset/trainset]")
    _build_split(ROOT / "cnntriset" / "trainset", TRAIN_WPMS, TRAIN_PER_CAT,
                 w_train, n_train, workers=args.workers)

    print("\n[cnntriset/valset]")
    _build_split(ROOT / "cnntriset" / "valset", EVAL_WPMS, EVAL_PER_CAT,
                 w_val, n_val, workers=args.workers)

    print("\n[cnntriset/testset]")
    _build_split(ROOT / "cnntriset" / "testset", EVAL_WPMS, EVAL_PER_CAT,
                 w_test, n_test, workers=args.workers)

    print("\nall sets generated.")

if __name__ == "__main__":
    main()
