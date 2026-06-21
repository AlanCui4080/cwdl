

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnnset import CnnSet, collate, IDX2CHAR
from model_v11 import CWModel, greedy_decode, CNN_T

ROOT = Path(__file__).parent
CKPT = ROOT / "checkpoints" / "best_v11.pt"
TEST_CSV = ROOT / "cnntriset" / "testset" / "index.csv"
TEST_ROOT = ROOT / "cnntriset" / "testset"
_SEQ_RE = re.compile(r"(\d+)_[^_]+_b(\d+)\.png$")

def _edit(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]

def load_meta(csv_path: Path) -> dict:

    meta = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = Path(row["path"]).parent.as_posix()
            m = _SEQ_RE.search(Path(row["path"]).name)
            seq = int(m.group(1)) if m else -1
            meta[(d, seq)] = (int(row["wpm"]), int(row["noise_db"]))
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "cer_heatmapv11.png"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    idx2char = ck.get("vocab", IDX2CHAR)
    model = CWModel(vocab_size=len(idx2char)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.ckpt}  epoch={ck.get('epoch')}  cer={ck.get('cer')}")

    ds = CnnSet(TEST_CSV, TEST_ROOT)
    meta = load_meta(TEST_CSV)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=4)

    sums = defaultdict(list)
    n_done = 0
    with torch.no_grad():
        for blocks, num_blocks, target, tlens, texts in loader:
            blocks = blocks.to(device)
            num_blocks = num_blocks.to(device)
            logits = model(blocks, num_blocks)
            ilens = num_blocks * CNN_T
            preds = greedy_decode(logits, ilens, idx2char)

            for i in range(len(texts)):
                p = ds.seqs[n_done][0][0]["path"]
                d = Path(p).parent.as_posix()
                m = _SEQ_RE.search(Path(p).name)
                sk = (d, int(m.group(1)) if m else -1)
                wpm, noise = meta[sk]

                ref = texts[i].upper()
                pred = preds[i]
                d_ed = _edit(ref, pred)
                cer = d_ed / max(1, len(ref))
                sums[(wpm, noise)].append(cer)
                n_done += 1

            if n_done % (args.batch_size * 20) == 0:
                print(f"  {n_done}/{len(ds)} done")

    wpms = sorted({k[0] for k in sums})
    noises = sorted({k[1] for k in sums})
    grid = np.full((len(noises), len(wpms)), np.nan)
    for (wpm, noise), vals in sums.items():
        grid[noises.index(noise), wpms.index(wpm)] = float(np.mean(vals))

    print("\nCER (mean)  rows=noise_db  cols=WPM")
    print("noise\\wpm " + " ".join(f"{w:>6d}" for w in wpms))
    for i, n in enumerate(noises):
        row = " ".join(
            f"{grid[i, j]:>6.3f}" if not np.isnan(grid[i, j]) else "   nan"
            for j in range(len(wpms)))
        print(f"{n:>9d} {row}")

    _plot(grid, wpms, noises, args.out)
    print(f"\nsaved heatmap -> {args.out}")

def _plot(grid, wpms, noises, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = 0.02
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   cmap="viridis_r", interpolation="nearest", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(wpms)))
    ax.set_xticklabels([str(w) for w in wpms])
    ax.set_yticks(range(len(noises)))
    ax.set_yticklabels([str(n) for n in noises])
    ax.set_xlabel("WPM")
    ax.set_ylabel("noise_db")
    ax.set_title("Testset CER (noise_db x WPM)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > vmax * 0.5 else "black"
            ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                    fontsize=6, color=color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CER")
    fig.tight_layout()
    fig.savefig(out, dpi=150)

if __name__ == "__main__":
    main()
