from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

VOCAB = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ") + ["<bos>", "<eos>", "<unk>"]
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
IDX2CHAR = {i + 1: c for i, c in enumerate(VOCAB)}
PAD = 0
BOS_IDX = CHAR2IDX["<bos>"]
EOS_IDX = CHAR2IDX["<eos>"]


import re

_token_re = re.compile(r"<bos>|<eos>|<unk>|.", re.IGNORECASE)


def encode(text: str) -> list[int]:
    ids = []
    for tok in _token_re.findall(text):
        if tok in CHAR2IDX:
            ids.append(CHAR2IDX[tok])
        else:
            ids.append(CHAR2IDX.get(tok.upper(), CHAR2IDX["<unk>"]))
    return ids


def load_sequences(index_csv: str | Path, img_root: str | Path):
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
        arr = np.load(self.img_root / rel)
        blocks = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(1)
        ids = encode(text)
        tgt_in = torch.tensor(ids[:-1], dtype=torch.long)
        tgt_out = torch.tensor(ids[1:], dtype=torch.long)
        return blocks, tgt_in, tgt_out, text


def collate(batch):
    blocks_list, tgt_in_list, tgt_out_list, texts = zip(*batch)
    Kmax = max(b.shape[0] for b in blocks_list)
    B = len(batch)
    blocks = torch.zeros(B, Kmax, 1, 16, 128, dtype=torch.float32)
    num_blocks = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(blocks_list):
        k = b.shape[0]
        blocks[i, :k] = b
        num_blocks[i] = k

    Tmax = max(t.size(0) for t in tgt_in_list)
    tgt_in = torch.full((B, Tmax), PAD, dtype=torch.long)
    tgt_out = torch.full((B, Tmax), PAD, dtype=torch.long)
    for i, (ti, to) in enumerate(zip(tgt_in_list, tgt_out_list)):
        L = ti.size(0)
        tgt_in[i, :L] = ti
        tgt_out[i, :L] = to

    return blocks, num_blocks, tgt_in, tgt_out, texts
