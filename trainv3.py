

from __future__ import annotations

from collections import defaultdict

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cnnsetv3 import CnnSetV3, collate, IDX2CHAR, encode
from modelv3 import CWModel, ctc_loss, greedy_decode, CNN_T
from tqdm import tqdm

print("IDX2CHAR:", IDX2CHAR)
print("len(IDX2CHAR):", len(IDX2CHAR))

test_char = 'A'
encoded = encode(test_char)
if isinstance(encoded, list):
    encoded = encoded[0]
decoded_char = IDX2CHAR[encoded - 1] if encoded > 0 else '<blank>'
print(f"encode('A')={encoded}, IDX2CHAR[{encoded - 1}]='{decoded_char}'")

try:
    from tqdm import trange
except Exception:
    trange = range

ROOT = Path(__file__).parent
TRAIN_CSV = ROOT / "cnntriset" / "trainset" / "index.csv"
TRAIN_IMG = ROOT / "cnntriset" / "trainset"
VAL_CSV = ROOT / "cnntriset" / "valset" / "index.csv"
VAL_IMG = ROOT / "cnntriset" / "valset"
CKPT_DIR = ROOT / "checkpoints"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class IndexSubset(torch.utils.data.Dataset):
    """从一个 base Dataset 中按 indices 取子集,用于过滤验证集。"""
    def __init__(self, base, indices):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]

def _edit_distance(a: str, b: str) -> int:
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

def _batch_cer(logits, ilens, texts, idx2char):

    preds = greedy_decode(logits, ilens, idx2char)
    total_d, total_chars = 0, 0
    for p, t in zip(preds, texts):
        ref = t.upper()
        total_d += _edit_distance(p, ref)
        total_chars += max(1, len(ref))
    return total_d / total_chars

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_d, total_chars = 0, 0
    n = 0
    pbar = tqdm(loader, desc="val", leave=False, dynamic_ncols=True)
    for blocks, num_blocks, target, tlens, texts in pbar:
        blocks = blocks.to(device)
        logits = model(blocks, num_blocks)
        ilens = num_blocks * CNN_T
        preds = greedy_decode(logits, ilens, IDX2CHAR)
        for p, t in zip(preds, texts):
            ref = t.upper()
            total_d += _edit_distance(p, ref)
            total_chars += len(ref)
            n += 1
        pbar.set_postfix(CER=f"{total_d/max(1,total_chars):.3f}", n=n)
    return total_d / max(1, total_chars), n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_noise_range", type=float, nargs=2,
                    default=[-10.0, 10.0],
                    metavar=("LO", "HI"),
                    help="验证集只取 noise_db 在 [LO,HI] 内的样本作为 CER 来源")
    ap.add_argument("--val_wpm_range", type=int, nargs=2,
                    default=[25, 65],
                    metavar=("LO", "HI"),
                    help="验证集只取 wpm 在 [LO,HI] 内的样本")
    ap.add_argument("--val_stride", type=int, default=5,
                    help="验证集在 noise 过滤后, 每组内按 stride 均匀抽取")
    ap.add_argument("--logdir", type=str, default="runs")
    ap.add_argument("--no_tb", action="store_true")
    ap.add_argument("--resume", type=str, default=None,
                    help="checkpoint 路径, 恢复训练")
    ap.add_argument("--resume_lr", action="store_true",
                    help="resume 时一并恢复 scheduler/optimizer 的学习率状态; "
                         "默认仅恢复模型权重与训练进度, 学习率按新调度从头算")
    ap.add_argument("--warmup_epochs", type=int, default=0,
                    help="线性 warmup 的 epoch 数, 0 关闭")
    ap.add_argument("--patience", type=int, default=5,
                    help="连续若干 epoch 无更佳 checkpoint 则早停, 0 关闭")
    args = ap.parse_args()

    args.logdir = os.path.join(args.logdir, datetime.now().strftime("%Y%m%d_%H%M%S"))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_ds = CnnSetV3(TRAIN_CSV, TRAIN_IMG)
    print(f"train sequences: {len(train_ds)}")

    val_full = CnnSetV3(VAL_CSV, VAL_IMG)
    lo_n, hi_n = args.val_noise_range
    lo_w, hi_w = args.val_wpm_range
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, s in enumerate(val_full.seqs):
        _, _, wpm, noise = s
        if lo_n <= noise <= hi_n and lo_w <= wpm <= hi_w:
            groups[(wpm, noise)].append(i)
    val_idx: list[int] = []
    for key in sorted(groups):
        val_idx.extend(groups[key][::args.val_stride])
    val_ds = IndexSubset(val_full, val_idx)
    print(f"val sequences: {len(val_ds)} (noise_db in [{lo_n:g},{hi_n:g}], "
          f"wpm in [{lo_w},{hi_w}], stride={args.val_stride}, "
          f"from {len(val_full)} total)")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate,
                            pin_memory=True)

    model = CWModel(vocab_size=len(IDX2CHAR)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    # 多卡时自动用 DataParallel;保留裸 model 引用用于存取权重
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpu > 1:
        dp_model = nn.DataParallel(model)
        print(f"DataParallel on {n_gpu} GPUs")
    else:
        dp_model = model

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0 / args.warmup_epochs, end_factor=1.0,
            total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs - args.warmup_epochs)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    CKPT_DIR.mkdir(exist_ok=True)
    best_cer = float("inf")
    global_step = 0
    no_improve = 0
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck and args.resume_lr:
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            global_step = ck.get("global_step", 0)
            best_cer = ck.get("best_cer", float("inf"))
            no_improve = ck.get("no_improve", 0)
            start_epoch = ck.get("epoch", -1) + 1
        else:
            # 不加载 opt/sched 状态: 学习率由新建的 scheduler 从头排程,
            # 仅恢复训练进度 (start_epoch / global_step / best_cer)
            global_step = ck.get("global_step", 0)
            best_cer = ck.get("best_cer", float("inf"))
            no_improve = ck.get("no_improve", 0)
            start_epoch = ck.get("epoch", -1) + 1 if "epoch" in ck else 0
            if "opt" in ck:
                print(f"[resume] --resume_lr 未启用, 仅恢复模型权重与训练进度, "
                      f"学习率按新 scheduler 从头排程")
            else:
                print(f"[resume] 旧格式 checkpoint, 仅恢复模型权重 "
                      f"(epoch={ck.get('epoch')}, cer={ck.get('cer')})")
        print(f"[resume] {args.resume} -> start_epoch={start_epoch} "
              f"global_step={global_step} best_cer={best_cer:.4f} "
              f"no_improve={no_improve}")
    writer = None if args.no_tb else SummaryWriter(args.logdir)
    try:
        for epoch in trange(start_epoch, args.epochs, desc="epoch", dynamic_ncols=True):
            dp_model.train()
            pbar = tqdm(train_loader, desc=f"e{epoch:02d}", dynamic_ncols=True)
            running = 0.0
            rn = 0
            running_cer = 0.0
            rcn = 0
            for blocks, num_blocks, target, tlens, texts in pbar:
                blocks = blocks.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                tlens = tlens.to(device, non_blocking=True)

                logits = dp_model(blocks, num_blocks)
                ilens = (num_blocks * CNN_T).to(device)
                loss = ctc_loss(logits, target, ilens, tlens)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

                running += loss.item()
                rn += 1
                global_step += 1
                batch_cer = _batch_cer(logits, ilens, texts, IDX2CHAR)
                running_cer += batch_cer
                rcn += 1
                if writer:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", sched.get_last_lr()[0],
                                      global_step)
                    writer.add_scalar("train/cer", batch_cer, global_step)
                pbar.set_postfix(loss=f"{running/rn:.4f}",
                                 cer=f"{running_cer/rcn:.3f}",
                                 lr=f"{sched.get_last_lr()[0]:.1e}")
                if global_step % 100 == 0:
                    with torch.no_grad():
                        preds = greedy_decode(logits, ilens, IDX2CHAR)
                    tqdm.write(f"[e{epoch:02d} it{global_step}] "
                               f"loss={loss.item():.4f} cer={batch_cer:.3f}")
                    for t, p in zip(texts[:2], preds[:2]):
                        tqdm.write(f"  gt  : {t.upper()}")
                        tqdm.write(f"  pred: {p}")

            sched.step()
            cer, n = evaluate(dp_model, val_loader, device)
            tqdm.write(f"[epoch {epoch:02d}] val CER={cer:.4f} (n={n}) "
                       f"lr={sched.get_last_lr()[0]:.2e} best={best_cer:.4f}")

            if writer:
                writer.add_scalar("val/cer", cer, epoch)
                writer.flush()

            if cer < best_cer:
                best_cer = cer
                no_improve = 0
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "cer": cer, "vocab": IDX2CHAR,
                            "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "global_step": global_step, "best_cer": best_cer,
                            "no_improve": no_improve, "epochs": args.epochs},
                           CKPT_DIR / "best_v3.pt")
                tqdm.write(f"  -> saved best_v3.pt (CER={cer:.4f})")
            else:
                no_improve += 1
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "global_step": global_step, "best_cer": best_cer,
                        "no_improve": no_improve, "epochs": args.epochs,
                        "vocab": IDX2CHAR},
                       CKPT_DIR / "last_v3.pt")
            if args.patience > 0 and no_improve >= args.patience:
                tqdm.write(f"[early stop] {args.patience} epochs 无更佳 "
                           f"checkpoint, 停止于 epoch {epoch:02d} "
                           f"(best CER={best_cer:.4f})")
                break

    finally:
        if writer:
            writer.close()

    print(f"done. best CER={best_cer:.4f}")

if __name__ == "__main__":
    main()
