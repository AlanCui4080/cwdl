
from __future__ import annotations

from collections import defaultdict

import argparse
import os
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cnnsetv5 import CnnSetV5, collate, IDX2CHAR, encode
from modelv5 import CWModel, ctc_loss, greedy_decode, CNN_T
from tqdm import tqdm

print("IDX2CHAR:", IDX2CHAR)
print("len(IDX2CHAR):", len(IDX2CHAR))

test_char = 'A'
encoded = encode(test_char)
if isinstance(encoded, list):
    encoded = encoded[0]
decoded_char = IDX2CHAR[encoded] if encoded < len(IDX2CHAR) else '<oob>'
print(f"encode('A')={encoded}, IDX2CHAR[{encoded}]='{decoded_char}'")

# space-collapse: 评估时将参考文本的连续 space 折叠为单个,
# 与 encode() 的 target 折叠和 CTC greedy decode 的重复折叠对齐
_multi_space_re = re.compile(r" {2,}")


def _collapse_spaces(s: str) -> str:
    return _multi_space_re.sub(" ", s)

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
        ref = _collapse_spaces(t.upper())
        total_d += _edit_distance(p, ref)
        total_chars += max(1, len(ref))
    return total_d / total_chars

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=None):
    model.eval()
    total_d, total_chars = 0, 0
    n = 0
    pbar = tqdm(loader, desc="val", leave=False, dynamic_ncols=True)
    for blocks, num_blocks, target, tlens, texts in pbar:
        blocks = blocks.to(device)
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            logits = model(blocks, num_blocks)
        ilens = num_blocks * CNN_T
        preds = greedy_decode(logits, ilens, IDX2CHAR)
        for p, t in zip(preds, texts):
            ref = _collapse_spaces(t.upper())
            total_d += _edit_distance(p, ref)
            total_chars += len(ref)
            n += 1
        pbar.set_postfix(CER=f"{total_d/max(1,total_chars):.3f}", n=n)
    return total_d / max(1, total_chars), n

def _load_v3_to_v5(v3_path: str, v5_model: CWModel, device: torch.device):
    """net2net: 将 v3 checkpoint 的 CNN 权重迁移到 v5 模型。

    v3 CNN -> v5 CNN 层对应关系:
      cnn.0  (1->16)  -> cnn2d.0  (1->32)   输出通道复制 x2
      cnn.1  BN(16)   -> cnn2d.1  BN(32)    扩张
      cnn.3  (16->32) -> cnn2d.3  (32->32)  输入通道复制 x2, 权重 x0.5
      cnn.4  BN(32)   -> cnn2d.4  BN(32)    直接复制
      conv1d_1 (32->64) -> conv1d_1 (32->64) 直接复制
      conv1d_2 (64->64) -> conv1d_2 (64->64) 直接复制
      bn1/bn2  BN(64)  -> bn1/bn2  BN(64)    直接复制
      conv1d_3 (64->64) -> conv1d_3 (64->128) 输出通道复制 x2
      bn3  BN(64)      -> bn3  BN(128)        扩张
    LSTM/head 结构不同, 随机初始化。
    """
    ck = torch.load(v3_path, map_location=device, weights_only=False)
    v3sd = ck["model"]
    v5sd = v5_model.state_dict()
    copied = 0

    def _expand_out(w, factor=2):
        """输出通道扩张: 沿 dim=0 复制, 保持函数不变"""
        return w.repeat(factor, *([1] * (w.dim() - 1)))

    def _expand_in(w, factor=2):
        """输入通道扩张: 沿 dim=1 复制, 缩放 1/factor 保持输出幅度不变"""
        return w.repeat(1, factor, *([1] * (w.dim() - 2))) / factor

    def _expand_bn(sd_old, sd_new, prefix_old, prefix_new, factor=2):
        """BatchNorm 通道扩张: weight/bias/running_mean/running_var 复制"""
        for suffix in ["weight", "bias", "running_mean", "running_var"]:
            sd_new[f"{prefix_new}.{suffix}"] = sd_old[f"{prefix_old}.{suffix}"].repeat(factor)
        if f"{prefix_old}.num_batches_tracked" in sd_old:
            sd_new[f"{prefix_new}.num_batches_tracked"] = sd_old[f"{prefix_old}.num_batches_tracked"]

    # conv2d[0]: 1->16 -> 1->32 (输出扩张)
    v5sd["cnn2d.0.weight"] = _expand_out(v3sd["cnn.0.weight"])
    _expand_bn(v3sd, v5sd, "cnn.1", "cnn2d.1")
    copied += 2

    # conv2d[3]: 16->32 -> 32->32 (输入扩张, 权重 x0.5)
    v5sd["cnn2d.3.weight"] = _expand_in(v3sd["cnn.3.weight"])
    # bn cnn.4 -> cnn2d.4 (32ch, 直接复制)
    _expand_bn(v3sd, v5sd, "cnn.4", "cnn2d.4", factor=1)
    copied += 2

    # conv1d_1, conv1d_2: 直接复制
    v5sd["conv1d_1.weight"] = v3sd["conv1d_1.weight"]
    v5sd["conv1d_2.weight"] = v3sd["conv1d_2.weight"]
    _expand_bn(v3sd, v5sd, "bn1", "bn1", factor=1)
    _expand_bn(v3sd, v5sd, "bn2", "bn2", factor=1)
    copied += 4

    # conv1d_3: 64->64 -> 64->128 (输出扩张)
    v5sd["conv1d_3.weight"] = _expand_out(v3sd["conv1d_3.weight"])
    _expand_bn(v3sd, v5sd, "bn3", "bn3", factor=2)
    copied += 2

    v5_model.load_state_dict(v5sd)
    print(f"[pretrain_v3] 从 {v3_path} 迁移 {copied} 组 CNN 权重 "
          f"(epoch={ck.get('epoch')}, cer={ck.get('cer'):.4f})")
    print(f"[pretrain_v3] LSTM/head 保持随机初始化")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2,
                    help="AdamW weight decay (decoupled)")
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
    ap.add_argument("--logdir", type=str, default="runs_v5")
    ap.add_argument("--no_tb", action="store_true")
    ap.add_argument("--resume", type=str, default=None,
                    help="checkpoint 路径, 恢复训练")
    ap.add_argument("--pretrain_v3", type=str, default=None,
                    help="从 v3 checkpoint net2net 迁移 CNN 权重作为初始值 "
                         "(LSTM/head 随机初始化)")
    ap.add_argument("--resume_lr", action="store_true",
                    help="resume 时一并恢复 scheduler/optimizer 的学习率状态; "
                         "默认仅恢复模型权重与训练进度, 学习率按新调度从头算")
    ap.add_argument("--warmup_epochs", type=int, default=0,
                    help="线性 warmup 的 epoch 数, 0 关闭")
    ap.add_argument("--patience", type=int, default=5,
                    help="连续若干 epoch 无更佳 checkpoint 则早停, 0 关闭")
    ap.add_argument("--amp_dtype", type=str, default="bf16",
                    choices=["bf16", "fp16", "fp32"],
                    help="混合精度 dtype: bf16 (默认, 无需 scaler), "
                         "fp16 (需 GradScaler), fp32 (禁用)")
    args = ap.parse_args()

    args.logdir = os.path.join(args.logdir, datetime.now().strftime("%Y%m%d_%H%M%S"))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if args.amp_dtype == "fp32" or device.type != "cuda":
        amp_dtype = None
    else:
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_amp = amp_dtype is not None
    use_scaler = use_amp and amp_dtype == torch.float16  # bf16 不需要 scaler
    print(f"amp: {args.amp_dtype}  dtype={amp_dtype}  scaler={use_scaler}")

    train_ds = CnnSetV5(TRAIN_CSV, TRAIN_IMG)
    print(f"train sequences: {len(train_ds)}")

    val_full = CnnSetV5(VAL_CSV, VAL_IMG)
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

    model = CWModel(vocab_size=len(IDX2CHAR) - 1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    # 多卡时自动用 DataParallel;保留裸 model 引用用于存取权重
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpu > 1:
        dp_model = nn.DataParallel(model)
        print(f"DataParallel on {n_gpu} GPUs")
    else:
        dp_model = model

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # 指数衰减: 每 5 epoch 降 1 个数量级 (gamma = 10^(-1/5))
    # 到达 lr_floor (1e-6) 后改为线性衰减到 0
    lr_floor = 1e-6
    decay = 10.0 ** (-1.0 / 5.0)  # ~= 0.6310
    n_exp = max(1, int(round(
        (torch.log10(torch.tensor(args.lr)) -
         torch.log10(torch.tensor(lr_floor))).item() * 5.0)))
    schedulers = []
    milestones = []

    if args.warmup_epochs > 0:
        schedulers.append(torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0 / args.warmup_epochs, end_factor=1.0,
            total_iters=args.warmup_epochs))
        milestones.append(args.warmup_epochs)

    schedulers.append(torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay))
    milestones.append(milestones[-1] + n_exp if milestones else n_exp)

    # LinearLR base = optimizer 初始 lr, start_factor = lr_floor / args.lr
    # 从指数衰减结束时的 lr_floor 平滑续接到 0
    n_linear = max(1, args.epochs - milestones[-1])
    schedulers.append(torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=lr_floor / args.lr, end_factor=0.0,
        total_iters=n_linear))

    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=schedulers, milestones=milestones)
    print(f"optimizer: AdamW (wd={args.weight_decay})  "
          f"sched: warmup={args.warmup_epochs}ep -> "
          f"exp(gamma={decay:.4f}, {n_exp}ep to {lr_floor:.0e}) -> "
          f"linear({n_linear}ep to 0)")

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
            if "scaler" in ck and use_scaler:
                scaler.load_state_dict(ck["scaler"])
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
    if args.pretrain_v3 and not args.resume:
        _load_v3_to_v5(args.pretrain_v3, model, device)
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

                with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
                    logits = dp_model(blocks, num_blocks)
                    ilens = (num_blocks * CNN_T).to(device)
                    loss = ctc_loss(logits, target, ilens, tlens)

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(opt)
                scaler.update()

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
                        tqdm.write(f"  gt  : {_collapse_spaces(t.upper())}")
                        tqdm.write(f"  pred: {p}")

            sched.step()
            cer, n = evaluate(dp_model, val_loader, device, amp_dtype=amp_dtype)
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
                            "scaler": scaler.state_dict(),
                            "global_step": global_step, "best_cer": best_cer,
                            "no_improve": no_improve, "epochs": args.epochs},
                           CKPT_DIR / "best_v5.pt")
                tqdm.write(f"  -> saved best_v5.pt (CER={cer:.4f})")
            else:
                no_improve += 1
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(),
                        "global_step": global_step, "best_cer": best_cer,
                        "no_improve": no_improve, "epochs": args.epochs,
                        "vocab": IDX2CHAR},
                       CKPT_DIR / "last_v5.pt")
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
