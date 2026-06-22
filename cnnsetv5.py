from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

VOCAB = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .?=/") + ["[DEL]", "[SK]", "[BK]", "<unk>"]
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
IDX2CHAR = ["<blank>"] + VOCAB  # IDX2CHAR[0]=blank, IDX2CHAR[i]=vocab token at model output i
PAD = 0
BLANK = 0
SPACE_IDX = CHAR2IDX[" "]


import re

_token_re = re.compile(r"<unk>|\[DEL\]|\[SK\]|\[BK\]|.", re.IGNORECASE)


def encode(text: str) -> list[int]:
    ids = []
    for tok in _token_re.findall(text):
        if tok in CHAR2IDX:
            ids.append(CHAR2IDX[tok])
        else:
            ids.append(CHAR2IDX.get(tok.upper(), CHAR2IDX["<unk>"]))
    # 超长序列: 连续 space 折叠为单个, 使 CTC 对任意数量的连续 space 只需输出一个
    out: list[int] = []
    for i in ids:
        if i == SPACE_IDX and out and out[-1] == SPACE_IDX:
            continue
        out.append(i)
    return out


def load_sequences(index_csv: str | Path, img_root: str | Path):
    seqs: list[tuple[str, str, int, int]] = []
    with open(index_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seqs.append((row["path"], row["text"],
                         int(row["wpm"]), int(row["noise_db"])))
    return seqs


class CnnSetV5(Dataset):
    def __init__(self, index_csv: str | Path, img_root: str | Path):
        self.img_root = Path(img_root)
        self.seqs = load_sequences(index_csv, img_root)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        rel, text, _, _ = self.seqs[idx]
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
