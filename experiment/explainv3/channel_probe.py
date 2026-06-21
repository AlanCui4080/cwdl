"""通道探测: 枚举 0-3 字符长度的随机字母数字空格组合 (40wpm, 无噪声),
记录每个 conv1d_3 通道在各字符标签下的平均激活, 反推通道语义。

方法:
1. 生成大量短序列频谱图 (1-3 字符 + 空序列)
2. 模型前向, 取 conv1d_3 输出 (BK, 64, T)
3. 用模型自身 CTC 预测给每帧打标签 (argmax -> 字符或 blank)
4. 按字符标签聚合通道激活, 计算均值 -> 通道-字符调谐矩阵
5. 输出每通道 top-3 语义 + 热力图
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from genmorse import compute_morse
from spectrogram import compute_spectrogram, slice_blocks
from modelv3 import CWModel, greedy_decode, CNN_T
from cnnsetv3 import IDX2CHAR, CHAR2IDX

CKPT = ROOT / "checkpoints" / "bestv3.pt"
OUT = Path(__file__).resolve().parent
VOCAB = IDX2CHAR  # 37 chars
BLOCK_W = 128
BLOCK_HOP = 64
WPM = 40.0
NOISE_DB = 0.0
CENTER_FREQ = 1000.0


def gen_spec(text: str) -> np.ndarray:
    """生成 text 的频谱图块, 返回 (K,16,128) uint8。"""
    audio = compute_morse(text, wpm=WPM, noise_db=NOISE_DB)
    img = compute_spectrogram(audio, center_freq=CENTER_FREQ)
    blocks = slice_blocks(img, BLOCK_W, BLOCK_HOP)
    if not blocks:
        # 填一个空块
        return np.zeros((1, 16, 128), dtype=np.uint8)
    return np.stack([blk for _, _, blk in blocks], axis=0)


def gen_samples(n_per_len: int = 400) -> list[tuple[str, np.ndarray]]:
    """枚举 0-3 字符长度, 每长度随机生成 n_per_len 个样本。"""
    chars = [c for c in VOCAB if c != " "]
    out: list[tuple[str, np.ndarray]] = []
    for length in range(0, 4):
        for _ in range(n_per_len):
            if length == 0:
                text = ""
            else:
                text = "".join(np.random.choice(chars) for _ in range(length))
            spec = gen_spec(text)
            out.append((text, spec))
    return out


def main():
    device = torch.device("cpu")
    print(f"loading {CKPT}")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    vocab = ck.get("vocab", IDX2CHAR)
    model = CWModel(vocab_size=len(vocab))
    model.load_state_dict(ck["model"])
    model.eval()

    # char -> 通道激活累加器
    n_ch = 64  # conv1d_3 输出通道
    char_act_sum = defaultdict(lambda: np.zeros(n_ch, dtype=np.float64))
    char_act_cnt = defaultdict(int)

    # hook conv1d_3 (relu 后, 即 bn3 后的 relu 输出)
    # 实际上 conv1d_3 模块输出后还有 bn3+relu, hook relu 不方便,
    # 直接 hook conv1d_3 拿到卷积输出 (未过 bn/relu), 也能反映通道调谐
    feat_buf: list[torch.Tensor] = []

    def hook(_m, _i, o):
        feat_buf.append(o.detach())

    h = model.conv1d_3.register_forward_hook(hook)

    np.random.seed(42)
    samples = gen_samples(n_per_len=400)
    print(f"generated {len(samples)} samples (0-3 chars, 400/len)")

    for si, (text, spec) in enumerate(samples):
        K = spec.shape[0]
        blocks = torch.from_numpy(spec.astype(np.float32) / 255.0)
        blocks = blocks.unsqueeze(1).unsqueeze(0)  # (1,K,1,16,128)
        num_blocks = torch.tensor([K], dtype=torch.long)

        feat_buf.clear()
        with torch.no_grad():
            logits = model(blocks, num_blocks)
        feat = feat_buf[0]  # (K, 64, T=128)
        feat = feat.cpu().numpy()  # (K, 64, 128)

        ilens = (num_blocks * CNN_T).long()
        # 帧级标签: argmax
        L = int(ilens[0].item())
        argmax = logits[0, :L].argmax(-1).cpu().numpy()  # (L,)

        # 把帧标签对齐到通道激活: feat 是 (K,64,128), 拼成 (64, L)
        feat_flat = feat.transpose(1, 0, 2).reshape(n_ch, -1)  # (64, K*128)
        # argmax 长度 = K*128 = L
        for t in range(L):
            label = int(argmax[t])
            if label == 0:
                ch = "<blank>"
            else:
                ch = vocab[label - 1]
            char_act_sum[ch] += feat_flat[:, t]
            char_act_cnt[ch] += 1

        if (si + 1) % 200 == 0:
            print(f"  [{si+1}/{len(samples)}] processed")

    h.remove()

    # 构建调谐矩阵: (n_chars, n_ch)
    # 按 vocab 序排列: <blank>, 空格, A-Z, 0-9
    ordered = ["<blank>"] + list(vocab)
    chars_sorted = [c for c in ordered if c in char_act_cnt]
    n_chars = len(chars_sorted)
    matrix = np.zeros((n_chars, n_ch), dtype=np.float64)
    cnts = np.zeros(n_chars, dtype=np.float64)
    for i, c in enumerate(chars_sorted):
        matrix[i] = char_act_sum[c] / max(1, char_act_cnt[c])
        cnts[i] = char_act_cnt[c]

    # 标准化: 每通道 z-score, 便于跨通道比较
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True) + 1e-8
    matrix_z = (matrix - mean) / std

    # ---- 热力图 ----
    fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                             gridspec_kw={"width_ratios": [1, 1]})
    # 左: 原始均值
    im0 = axes[0].imshow(matrix, aspect="auto", cmap="magma")
    axes[0].set_yticks(range(n_chars))
    axes[0].set_yticklabels([f"{c}({int(n)})" for c, n in
                             zip(chars_sorted, cnts)], fontsize=7)
    axes[0].set_xlabel("conv1d_3 channel")
    axes[0].set_title("channel mean activation per char")
    fig.colorbar(im0, ax=axes[0], fraction=0.03)
    # 右: z-score
    im1 = axes[1].imshow(matrix_z, aspect="auto", cmap="RdBu_r",
                          vmin=-3, vmax=3)
    axes[1].set_yticks(range(n_chars))
    axes[1].set_yticklabels([f"{c}({int(n)})" for c, n in
                             zip(chars_sorted, cnts)], fontsize=7)
    axes[1].set_xlabel("conv1d_3 channel")
    axes[1].set_title("z-score: which channels fire for which char")
    fig.colorbar(im1, ax=axes[1], fraction=0.03)
    fig.suptitle("Channel probing (0-3 char, 40wpm, noiseless): "
                 "conv1d_3 channel -> char tuning", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "channel_probe_tuning.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved channel_probe_tuning.png")

    # ---- 每通道 top-3 语义 ----
    lines = ["conv1d_3 channel -> top-3 char (by z-score)", ""]
    for ch in range(n_ch):
        scores = matrix_z[:, ch]
        top = np.argsort(-scores)[:3]
        top_str = ", ".join(
            f"{chars_sorted[i]}({scores[i]:+.2f}|n={int(cnts[i])})"
            for i in top)
        lines.append(f"ch{ch:02d}: {top_str}")
    (OUT / "channel_probe_top.txt").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"  saved channel_probe_top.txt")

    # ---- 每字符 top-3 通道 (反向) ----
    lines = ["", "char -> top-3 conv1d_3 channel (by z-score)", ""]
    for ci, c in enumerate(chars_sorted):
        scores = matrix_z[ci]
        top = np.argsort(-scores)[:3]
        top_str = ", ".join(f"ch{int(i):02d}({scores[i]:+.2f})" for i in top)
        lines.append(f"{c:8s}(n={int(cnts[ci]):5d}): {top_str}")
    with open(OUT / "channel_probe_top.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\ndone. outputs in", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
