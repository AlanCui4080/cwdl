

from __future__ import annotations

from pathlib import Path

import torch

from cnnset import CnnSet, collate, IDX2CHAR
from model import CWModel, greedy_decode, CNN_T

ROOT = Path(__file__).parent
CKPT = ROOT / "checkpoints" / "best.pt"

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    idx2char = ck.get("vocab", IDX2CHAR)
    model = CWModel(vocab_size=len(idx2char)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {CKPT.name}  epoch={ck.get('epoch')}  cer={ck.get('cer')}")

    for name, csv, img_root in [
        ("TRAIN", ROOT / "cnntriset" / "trainset" / "index.csv", ROOT / "cnntriset" / "trainset"),
        ("TEST ", ROOT / "cnntriset" / "testset" / "index.csv", ROOT / "cnntriset" / "testset"),
    ]:
        ds = CnnSet(csv, img_root)
        sample = [ds[0]]
        blocks, num_blocks, target, tlens, texts = collate(sample)
        blocks = blocks.to(device)
        num_blocks = num_blocks.to(device)

        with torch.no_grad():
            logits = model(blocks, num_blocks)
            ilens = num_blocks * CNN_T
            preds = greedy_decode(logits, ilens, idx2char)

        ref = texts[0].upper()
        pred = preds[0]
        d = _edit(ref, pred)
        cer = d / max(1, len(ref))

        print(f"\n[{name}] blocks={int(num_blocks[0])} target_len={int(tlens[0])}")
        print(f"  ref : {ref}")
        print(f"  pred: {pred}")
        print(f"  CER : {cer:.3f}  ({d}/{len(ref)})")

if __name__ == "__main__":
    main()
