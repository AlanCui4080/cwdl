"""按莫尔斯电码模式分类的通道探测。

把每个字符的莫尔斯码 (点划序列) 作为分类键, 统计 conv1d_3 通道对
各电码模式的响应, 直观揭示通道学到的电码结构特征。
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

from genmorse import compute_morse, MORSE_MAP
from spectrogram import compute_spectrogram, slice_blocks
from modelv3 import CWModel, CNN_T
from cnnsetv3 import IDX2CHAR

CKPT = ROOT / "checkpoints" / "bestv3.pt"
OUT = Path(__file__).resolve().parent
BLOCK_W = 128
BLOCK_HOP = 64
WPM = 40.0
NOISE_DB = 0.0
CENTER_FREQ = 1000.0
N_CH = 64


def gen_spec(text: str) -> np.ndarray:
    audio = compute_morse(text, wpm=WPM, noise_db=NOISE_DB)
    img = compute_spectrogram(audio, center_freq=CENTER_FREQ)
    blocks = slice_blocks(img, BLOCK_W, BLOCK_HOP)
    if not blocks:
        return np.zeros((1, 16, 128), dtype=np.uint8)
    return np.stack([blk for _, _, blk in blocks], axis=0)


def morse_sort_key(code: str) -> tuple:
    """排序: 先按长度, 再按点(dit<dash)字典序。"""
    return (len(code), code.replace(".", "0").replace("-", "1"))


def main():
    device = torch.device("cpu")
    print(f"loading {CKPT}")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    vocab = ck.get("vocab", IDX2CHAR)
    model = CWModel(vocab_size=len(vocab))
    model.load_state_dict(ck["model"])
    model.eval()

    # 建立 char -> morse 映射 (只用 vocab 内的字符)
    char_morse = {}
    for c in vocab:
        if c.upper() in MORSE_MAP:
            char_morse[c] = MORSE_MAP[c.upper()]
    # 空格特殊: 莫尔斯用词间分隔, 这里记为 "/"
    if " " in vocab and " " not in char_morse:
        char_morse[" "] = "/"

    print(f"chars with morse: {len(char_morse)}")
    for c, m in sorted(char_morse.items(), key=lambda x: morse_sort_key(x[1])):
        print(f"  {c!r:4s} -> {m}")

    # 每个字符重复生成 N 次, 收集 conv1d_3 激活
    N_REP = 60
    feat_buf: list[torch.Tensor] = []

    def hook(_m, _i, o):
        feat_buf.append(o.detach())

    h = model.conv1d_3.register_forward_hook(hook)

    # char -> 激活累加 (只取有信号帧, 排除全零帧)
    char_act_sum: dict[str, np.ndarray] = {}
    char_act_cnt: dict[str, int] = {}

    np.random.seed(42)
    for ci, (ch, morse) in enumerate(sorted(char_morse.items(),
                                            key=lambda x: morse_sort_key(x[1]))):
        acc = np.zeros(N_CH, dtype=np.float64)
        cnt = 0
        for _ in range(N_REP):
            spec = gen_spec(ch)
            K = spec.shape[0]
            blocks = torch.from_numpy(spec.astype(np.float32) / 255.0)
            blocks = blocks.unsqueeze(1).unsqueeze(0)
            num_blocks = torch.tensor([K], dtype=torch.long)

            feat_buf.clear()
            with torch.no_grad():
                model(blocks, num_blocks)
            feat = feat_buf[0].cpu().numpy()  # (K, 64, 128)
            # 取所有帧的激活均值 (跨 K 块和 128 时间步)
            acc += feat.mean(axis=(0, 2))
            cnt += 1
        char_act_sum[ch] = acc / cnt
        char_act_cnt[ch] = cnt
        print(f"  [{ci+1}/{len(char_morse)}] {ch!r} ({morse}) done")

    h.remove()

    # 构建矩阵: (n_chars, n_ch), 按电码模式排序
    chars_ordered = [c for c, _ in sorted(char_morse.items(),
                                          key=lambda x: morse_sort_key(x[1]))]
    n_chars = len(chars_ordered)
    matrix = np.zeros((n_chars, N_CH), dtype=np.float64)
    for i, c in enumerate(chars_ordered):
        matrix[i] = char_act_sum[c]

    # z-score 标准化 (每通道)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True) + 1e-8
    matrix_z = (matrix - mean) / std

    # 标签: "A(.-)" 格式
    ylabels = [f"{c}({char_morse[c]})" for c in chars_ordered]

    # ---- 热力图: 字符(电码) x 通道 ----
    fig, axes = plt.subplots(1, 2, figsize=(22, 12),
                             gridspec_kw={"width_ratios": [1, 1]})
    im0 = axes[0].imshow(matrix, aspect="auto", cmap="magma")
    axes[0].set_yticks(range(n_chars))
    axes[0].set_yticklabels(ylabels, fontsize=8, family="monospace")
    axes[0].set_xlabel("conv1d_3 channel")
    axes[0].set_title("mean activation")
    fig.colorbar(im0, ax=axes[0], fraction=0.03)

    im1 = axes[1].imshow(matrix_z, aspect="auto", cmap="RdBu_r",
                          vmin=-3, vmax=3)
    axes[1].set_yticks(range(n_chars))
    axes[1].set_yticklabels(ylabels, fontsize=8, family="monospace")
    axes[1].set_xlabel("conv1d_3 channel")
    axes[1].set_title("z-score: channel tuning per morse pattern")
    fig.colorbar(im1, ax=axes[1], fraction=0.03)

    fig.suptitle(f"Channel probing by morse pattern (40wpm, noiseless, "
                 f"N={N_REP}/char)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "morse_pattern_tuning.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved morse_pattern_tuning.png")

    # ---- 按电码长度分组聚合 ----
    len_groups: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(chars_ordered):
        m = char_morse[c]
        if m == "/":
            length = 0
        else:
            length = len(m)
        len_groups[length].append(i)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()
    for ax_idx, (length, idxs) in enumerate(sorted(len_groups.items())):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        sub = matrix_z[idxs]  # (n_in_group, n_ch)
        ax.imshow(sub, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_yticks(range(len(idxs)))
        ax.set_yticklabels([ylabels[i] for i in idxs], fontsize=7,
                           family="monospace")
        ax.set_xlabel("channel")
        label = f"len={length}" if length > 0 else "space"
        ax.set_title(f"{label} ({len(idxs)} chars)")
    for j in range(len(len_groups), len(axes)):
        axes[j].axis("off")
    fig.suptitle("z-score by morse code length group", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "morse_pattern_by_length.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved morse_pattern_by_length.png")

    # ---- 每通道 top-3 电码模式 ----
    lines = ["conv1d_3 channel -> top-3 morse pattern (by z-score)", ""]
    for ch in range(N_CH):
        scores = matrix_z[:, ch]
        top = np.argsort(-scores)[:3]
        top_str = ", ".join(
            f"{ylabels[i]}({scores[i]:+.2f})" for i in top)
        lines.append(f"ch{ch:02d}: {top_str}")
    (OUT / "morse_pattern_top.txt").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"  saved morse_pattern_top.txt")

    print("\ndone. outputs in", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
