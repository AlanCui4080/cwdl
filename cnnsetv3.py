

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

VOCAB = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
IDX2CHAR = VOCAB
BLANK = 0

def encode(text: str) -> list[int]:
    return [CHAR2IDX[c] for c in text.upper() if c in CHAR2IDX]

def load_sequences(index_csv: str | Path, img_root: str | Path):

    # 每 sequence 一个 .npy (uint8 [K,16,128]) + 一行 CSV,无需按 block 分组
    seqs: list[tuple[str, str, int, int]] = []
    with open(index_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seqs.append((row["path"], row["text"],
                         int(row["wpm"]), int(row["noise_db"])))
    return seqs

class CnnSetV3(Dataset):
    def __init__(self, index_csv: str | Path, img_root: str | Path):
        self.img_root = Path(img_root)
        self.seqs = load_sequences(index_csv, img_root)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        rel, text, _, _ = self.seqs[idx]
        # 直接读 npy,无 PNG 解码;uint8->float32 /255
        arr = np.load(self.img_root / rel)
        blocks = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(1)
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
