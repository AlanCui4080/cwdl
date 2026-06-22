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

from cnnsetv4 import CnnSetV3, collate, IDX2CHAR, BOS_IDX, EOS_IDX, PAD, encode
from modelv4 import CWModel, ce_loss, greedy_decode, CNN_T
from tqdm import tqdm

print("IDX2CHAR:", IDX2CHAR)
print("BOS_IDX:", BOS_IDX, "EOS_IDX:", EOS_IDX, "PAD:", PAD)

print("--- encoder/decoder test ---")
_test_texts = [
    "<bos>HELLO WORLD<eos>",
    "<bos>7IBT129PO17LX26TEUHNLI2IID1 CR8FITM4<eos>",
    "<bos>A B C 1 2 3<eos>",
    "<bos>UNK@#$<eos>",
]
for _tt in _test_texts:
    _ids = encode(_tt)
    _decoded = " ".join(IDX2CHAR.get(i, "?") for i in _ids)
    print(f"  text: {_tt}")
    print(f"  ids : {_ids}")
    print(f"  dec : {_decoded}")
    print(f"  tgt_in : {[IDX2CHAR.get(i,'?') for i in _ids[:-1]]}")
    print(f"  tgt_out: {[IDX2CHAR.get(i,'?') for i in _ids[1:]]}")
    print()
print("--- encoder/decoder test done ---")

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


@torch.no_grad()
def evaluate(model, loader, device):
    """评估 CER，model 应为未包装的原始模型（已在 eval 模式）"""
    model.eval()
    total_d, total_chars = 0, 0
    n = 0
    pbar = tqdm(loader, desc="val", leave=False, dynamic_ncols=True)
    for blocks, num_blocks, tgt_in, tgt_out, texts in pbar:
        blocks = blocks.to(device)
        # 直接调用未包装的模型进行贪婪解码
        pred_ids = greedy_decode(model, blocks, num_blocks, BOS_IDX, EOS_IDX)
        for pred_list, t in zip(pred_ids, texts):
            # FIX: 预测串也去掉特殊 token（BOS/EOS/PAD），与参考串对齐
            pred_str = "".join(
                IDX2CHAR.get(i, "") for i in pred_list
                if i not in (BOS_IDX, EOS_IDX, PAD)
            )
            ref_ids = encode(t)
            ref = "".join(
                IDX2CHAR.get(i, "") for i in ref_ids
                if i not in (BOS_IDX, EOS_IDX)
            )
            total_d += _edit_distance(pred_str, ref)
            total_chars += max(1, len(ref))
            n += 1
        pbar.set_postfix(CER=f"{total_d / max(1, total_chars):.3f}", n=n)
    return total_d / max(1, total_chars), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_noise_range", type=float, nargs=2,
                    default=[-10.0, 10.0],
                    metavar=("LO", "HI"))
    ap.add_argument("--val_wpm_range", type=int, nargs=2,
                    default=[25, 65],
                    metavar=("LO", "HI"))
    ap.add_argument("--val_stride", type=int, default=5)
    ap.add_argument("--logdir", type=str, default="runs_v4")
    ap.add_argument("--no_tb", action="store_true")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--resume_lr", action="store_true")
    ap.add_argument("--warmup_epochs", type=int, default=0)
    ap.add_argument("--patience", type=int, default=5)
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

    model = CWModel(vocab_size=len(IDX2CHAR) + 1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")

    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpu > 1:
        dp_model = nn.DataParallel(model)
        print(f"DataParallel on {n_gpu} GPUs")
    else:
        dp_model = model

    # 用于评估和调试打印的推理模型（去掉 DataParallel 包装）
    inference_model = model.module if hasattr(model, 'module') else model

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
            for blocks, num_blocks, tgt_in, tgt_out, texts in pbar:
                blocks = blocks.to(device, non_blocking=True)
                tgt_in = tgt_in.to(device, non_blocking=True)
                tgt_out = tgt_out.to(device, non_blocking=True)

                # MODIFIED: 直接 FP32 前向传播，不再使用 torch.autocast
                logits = dp_model(blocks, num_blocks, tgt_in)
                loss = ce_loss(logits, tgt_out, ignore_index=PAD,
                               bos_idx=BOS_IDX, eos_idx=EOS_IDX)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

                running += loss.item()
                rn += 1
                global_step += 1
                if writer:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", sched.get_last_lr()[0],
                                      global_step)
                pbar.set_postfix(loss=f"{running / rn:.4f}",
                                 lr=f"{sched.get_last_lr()[0]:.1e}")
                if global_step % 100 == 0:
                    # FIX: 切换为推理模型并 eval 模式，避免 dropout 干扰打印
                    inference_model.eval()
                    with torch.no_grad():
                        pred_ids = greedy_decode(inference_model, blocks,
                                                 num_blocks, BOS_IDX, EOS_IDX)
                    # 重新切回训练模式（对整个 model 生效）
                    dp_model.train()
                    tqdm.write(f"[e{epoch:02d} it{global_step}] "
                               f"loss={loss.item():.4f}")
                    for t, p in zip(texts[:2], pred_ids[:2]):
                        gt_ids = encode(t)
                        gt_content = [i for i in gt_ids if i not in (BOS_IDX, EOS_IDX)]
                        gt_str = "".join(IDX2CHAR.get(i, "") for i in gt_content)
                        # FIX: 预测串同样去除特殊 token
                        pred_str = "".join(
                            IDX2CHAR.get(i, "") for i in p
                            if i not in (BOS_IDX, EOS_IDX, PAD)
                        )
                        tqdm.write(f"  gt  : {gt_str}  ")
                        tqdm.write(f"  pred: {pred_str}")

            sched.step()
            # FIX: 评估时传入去包装后的推理模型，并在 evaluate 内部切换 eval
            cer, n = evaluate(inference_model, val_loader, device)
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
                           CKPT_DIR / "best_v4.pt")
                tqdm.write(f"  -> saved best_v4.pt (CER={cer:.4f})")
            else:
                no_improve += 1
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "global_step": global_step, "best_cer": best_cer,
                        "no_improve": no_improve, "epochs": args.epochs,
                        "vocab": IDX2CHAR},
                       CKPT_DIR / "last_v4.pt")
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
