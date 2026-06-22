
from __future__ import annotations

import argparse
import os

from builddatasetv3 import (
    RANDOM50_TXT,
    ROOT,
    EVAL_PER_CAT,
    NOISE_DB,
    POOL_SPLIT,
    _build_split,
    _load_tagged,
    _split_pool,
)

# 测试集 WPM 范围: 10-100, 步进 5
TEST_WPMS = tuple(range(10, 101, 5))

# 原 EVAL_PER_CAT=300 的 1/5
PER_CAT = EVAL_PER_CAT // 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0,
                    help="并行进程数 (0=自动, 默认 CPU 核数)")
    args = ap.parse_args()

    pool = _load_tagged(RANDOM50_TXT, "random50")
    print(f"random50 pool: {len(pool)}")

    _, _, p_test = _split_pool(pool, POOL_SPLIT)
    print(f"random50 split: test={len(p_test)}")

    print(f"workers: {args.workers if args.workers > 0 else os.cpu_count()}")
    print(f"NOISE_DB: {NOISE_DB}")
    print(f"TEST_WPMS: {TEST_WPMS}")

    print("\n[cnntriset/testset]")
    _build_split(ROOT / "cnntriset" / "testset", TEST_WPMS, PER_CAT,
                 p_test, workers=args.workers)

    print("\ntestset generated.")


if __name__ == "__main__":
    main()
