

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

VOCAB = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
IDX2CHAR = VOCAB
BLANK = 0

_SEQ_RE = re.compile(r"(\d+)_[^_]+_b(\d+)\.png$")

def encode(text: str) -> list[int]:
    return [CHAR2IDX[c] for c in text.upper() if c in CHAR2IDX]

def _seq_key(path: str) -> tuple[str, int]:

    d = Path(path).parent.as_posix()
    m = _SEQ_RE.search(Path(path).name)
    seq = int(m.group(1)) if m else -1
    return (d, seq)

def load_sequences(index_csv: str | Path, img_root: str | Path):

    img_root = Path(img_root)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    with open(index_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[_seq_key(row["path"])].append(row)
    seqs = []
    for rows in groups.values():
        rows.sort(key=lambda r: int(r["block"]))
        seqs.append((rows, rows[0]["text"]))
    return seqs

def _load_block(img_root: Path, rel: str) -> np.ndarray:
    arr = np.array(Image.open(img_root / rel), dtype=np.float32) / 255.0
    if arr.shape != (16, 128):
        pad_h = 16 - arr.shape[0]
        arr = np.pad(arr, ((0, max(0, pad_h)),
                            (0, max(0, 128 - arr.shape[1]))))
        arr = arr[:16, :128]
    return arr

class CnnSet(Dataset):
    def __init__(self, index_csv: str | Path, img_root: str | Path):
        self.img_root = Path(img_root)
        self.seqs = load_sequences(index_csv, img_root)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        rows, text = self.seqs[idx]
        blocks = np.stack(
            [_load_block(self.img_root, r["path"]) for r in rows], axis=0)
        blocks = torch.from_numpy(blocks).unsqueeze(1)
        target = torch.tensor(encode(text), dtype=torch.long)
        return blocks, target, text

def collate(batch):

    blocks_list, targets, texts = zip(*batch)
    Kmax = max(b.shape[0] for b in blocks_list)
    B = len(batch)
    blocks = torch.zeros(B, Kmax, 1, 16, 128, dtype=torch.float32)
    num_blocks = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(blocks_list):
        k = b.shape[0]
        blocks[i, :k] = b
        num_blocks[i] = k
    target = torch.cat(targets) if targets else torch.zeros(0, dtype=torch.long)
    target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)
    return blocks, num_blocks, target, target_lengths, texts
