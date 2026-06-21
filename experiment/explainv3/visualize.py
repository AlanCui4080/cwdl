from __future__ import annotations

import sys
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from modelv3 import CWModel, greedy_decode, CNN_T
from cnnsetv3 import IDX2CHAR

CKPT = ROOT / "checkpoints" / "bestv3.pt"
VAL_CSV = ROOT / "cnntriset" / "valset" / "index.csv"
VAL_IMG = ROOT / "cnntriset" / "valset"
OUT = Path(__file__).resolve().parent


def minmax(t: torch.Tensor) -> torch.Tensor:
    t = t.float()
    lo, hi = t.min(), t.max()
    return (t - lo) / (hi - lo + 1e-8)


def grid_dims(n: int) -> tuple[int, int]:
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def save_fig(fig, name: str):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def viz_conv2d_first(weight: torch.Tensor, name: str):
    """weight: (out, in=1, 3, 3). 经典 4x4 网格, 每格一个 3x3 滤波器."""
    out = weight.shape[0]
    rows, cols = grid_dims(out)
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < out:
            ax.imshow(weight[i, 0].detach().cpu().numpy(), cmap="viridis")
            ax.set_title(str(i), fontsize=6)
    fig.suptitle(f"{name}: {out} filters, each 1x3x3", fontsize=10)
    save_fig(fig, f"kernels_{name}.png")


def viz_conv2d_multi(weight: torch.Tensor, name: str):
    """weight: (out, in, 3, 3). 每个 out 滤波器画成 in 通道的子网格."""
    out_c, in_c, kh, kw = weight.shape
    in_rows, in_cols = grid_dims(in_c)
    gap = 1
    canvas_h = in_rows * (kh + gap)
    canvas_w = in_cols * (kw + gap)
    rows, cols = grid_dims(out_c)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i >= out_c:
            continue
        canvas = np.full((canvas_h, canvas_w), np.nan)
        for c in range(in_c):
            r0, c0 = divmod(c, in_cols)
            y0 = r0 * (kh + gap)
            x0 = c0 * (kw + gap)
            canvas[y0:y0 + kh, x0:x0 + kw] = weight[i, c].detach().cpu().numpy()
        ax.imshow(canvas, cmap="viridis")
        ax.set_title(str(i), fontsize=6)
    fig.suptitle(f"{name}: {out_c} filters, each {in_c}x{kh}x{kw}", fontsize=10)
    save_fig(fig, f"kernels_{name}.png")

    flat = weight.reshape(out_c, in_c * kh * kw).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, max(6, out_c * 0.12)))
    im = ax.imshow(flat, aspect="auto", cmap="seismic",
                   vmin=-np.abs(flat).max(), vmax=np.abs(flat).max())
    ax.set_ylabel("output channel")
    ax.set_xlabel("input channel x kernel pos")
    ax.set_title(f"{name} weight heatmap (out x in*{kh*kw})")
    fig.colorbar(im, ax=ax, fraction=0.02)
    save_fig(fig, f"kernels_{name}_heatmap.png")


def viz_conv1d(weight: torch.Tensor, name: str):
    """weight: (out, in, k). 画 out 个滤波器网格 + 全局 heatmap."""
    out_c, in_c, k = weight.shape
    rows, cols = grid_dims(out_c)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.2))
    axes = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i >= out_c:
            continue
        ax.imshow(weight[i].detach().cpu().numpy(), aspect="auto", cmap="viridis")
        ax.set_title(str(i), fontsize=6)
    fig.suptitle(f"{name}: {out_c} filters, each {in_c}x{k}", fontsize=10)
    save_fig(fig, f"kernels_{name}.png")

    flat = weight.reshape(out_c, in_c * k).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, max(6, out_c * 0.1)))
    im = ax.imshow(flat, aspect="auto", cmap="seismic",
                   vmin=-np.abs(flat).max(), vmax=np.abs(flat).max())
    ax.set_ylabel("output channel")
    ax.set_xlabel("input channel x kernel tap")
    ax.set_title(f"{name} weight heatmap (out x in*{k})")
    fig.colorbar(im, ax=ax, fraction=0.02)
    save_fig(fig, f"kernels_{name}_heatmap.png")


def feat_grid(act: torch.Tensor, title: str, fname: str):
    """通用特征图: 输出 (..., C) 把末维当通道, 其余当时间, 画通道-时间热力图."""
    a = act[0].detach().cpu().float()
    # 把任意前导维合并为时间, 最后一维为通道
    mat = a.reshape(-1, a.shape[-1]).numpy()
    fig, ax = plt.subplots(figsize=(14, max(4, mat.shape[1] * 0.05)))
    im = ax.imshow(mat.T, aspect="auto", cmap="magma",
                   vmin=np.percentile(mat, 1), vmax=np.percentile(mat, 99))
    ax.set_xlabel(f"steps = {mat.shape[0]}")
    ax.set_ylabel("channel")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02)
    save_fig(fig, fname)


def viz_small_compare(model, vocab, pick):
    """小样本横向对比: 输入 spectrogram 与每块预测字符上下对齐 (横排展开)。

    每个块是一个 16x128 的时频图, 横向首尾相接排成一条长带; 下方按块中心
    标注预测字符, 与原图块一一对应。
    """
    arr = np.load(VAL_IMG / pick["path"]).astype(np.float32) / 255.0
    blocks = torch.from_numpy(arr).unsqueeze(1).unsqueeze(0)
    num_blocks = torch.tensor([blocks.shape[1]], dtype=torch.long)
    K = blocks.shape[1]

    with torch.no_grad():
        logits = model(blocks, num_blocks)
    ilens = (num_blocks * CNN_T).long()
    pred = greedy_decode(logits, ilens, vocab)[0]

    L = int(ilens[0].item())
    argmax = logits[0, :L].argmax(-1).cpu().numpy()
    block_chars: list[str] = []
    for b in range(K):
        seg = argmax[b * CNN_T:(b + 1) * CNN_T]
        seen: list[int] = []
        for i in seg:
            i = int(i)
            if i == 0:
                continue
            if not seen or seen[-1] != i:
                seen.append(i)
        ch = "".join(vocab[i - 1] for i in seen) if seen else "_"
        block_chars.append(ch)

    inp = blocks[0, :, 0].numpy()  # (K,16,128)
    strip = np.concatenate([inp[b] for b in range(K)], axis=1)  # (16, K*128)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, K * 1.0), 4),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)
    axes[0].imshow(strip, aspect="auto", cmap="gray", origin="lower",
                   extent=[0, K * 128, 0, 16])
    for b in range(1, K):
        axes[0].axvline(b * 128, color="cyan", lw=0.5, alpha=0.6)
    axes[0].set_ylabel("freq bin")
    axes[0].set_title(f"small sample  K={K}  gt={pick['text']!r}  pred={pred!r}")
    axes[0].set_yticks([])

    ax1 = axes[1]
    ax1.set_xlim(0, K * 128)
    ax1.axis("off")
    for b in range(K):
        cx = (b + 0.5) * 128
        ax1.text(cx, 0.5, block_chars[b], ha="center", va="center",
                 fontsize=9, family="monospace",
                 color="red" if block_chars[b] == "_" else "black")
        if b > 0:
            ax1.axvline(b * 128, color="gray", lw=0.3, alpha=0.5)
    ax1.set_title("per-block predicted chars", fontsize=9, loc="left")

    save_fig(fig, "small_compare.png")


def viz_feature_pipeline(model, vocab, pick):
    """特征流水线: 展示 2D CNN 如何把 16px 高的输入压缩到 1D, 再由 1D CNN 提取时序特征。

    纵向排列各阶段, 标注张量形状, 直观看到高度 16->4->1 的压缩过程。
    """
    arr = np.load(VAL_IMG / pick["path"]).astype(np.float32) / 255.0
    K = min(blocks_shape := arr.shape[0], 4)  # 最多取 4 个块, 控制宽度
    arr = arr[:K]
    blocks = torch.from_numpy(arr).unsqueeze(1).unsqueeze(0)
    num_blocks = torch.tensor([K], dtype=torch.long)

    # hook 各阶段输出
    # cnn Sequential: [conv2d_0, bn, relu, conv2d_3, bn, relu]
    stage_acts: dict[str, torch.Tensor] = {}

    def hook(key):
        def fn(_m, _i, o):
            stage_acts[key] = o.detach()
        return fn

    handles = [
        model.cnn[2].register_forward_hook(hook("conv2d_0_relu")),  # after 1st relu
        model.cnn[5].register_forward_hook(hook("conv2d_3_relu")),  # after 2nd relu
        model.conv1d_1.register_forward_hook(hook("conv1d_1")),
        model.conv1d_2.register_forward_hook(hook("conv1d_2")),
        model.conv1d_3.register_forward_hook(hook("conv1d_3")),
    ]
    with torch.no_grad():
        model(blocks, num_blocks)
    for h in handles:
        h.remove()

    inp = arr  # (K, 16, 128)
    c0 = stage_acts["conv2d_0_relu"]     # (K, 16, 4, 128)
    c1 = stage_acts["conv2d_3_relu"]     # (K, 32, 1, 128)
    d1 = stage_acts["conv1d_1"]          # (K, 64, 128)
    d2 = stage_acts["conv1d_2"]          # (K, 64, 128)
    d3 = stage_acts["conv1d_3"]          # (K, 64, 128)

    # 拼接 K 个块的时间轴: 各块沿 width=128 拼接
    W = K * 128
    inp_strip = np.concatenate([inp[b] for b in range(K)], axis=1)  # (16, W)
    # conv2d_0: (K,16,4,128) -> 每块展平 (16*4, 128) -> 跨块拼 (64, W)
    c0_strip = np.concatenate([c0[b].reshape(16 * 4, 128) for b in range(K)],
                              axis=1)  # (64, W)
    # conv2d_3: (K,32,1,128) -> 跨块拼 (32, W)
    c1_strip = np.concatenate([c1[b, :, 0, :] for b in range(K)], axis=1)  # (32, W)
    d1_strip = np.concatenate([d1[b] for b in range(K)], axis=1)  # (64, W)
    d2_strip = np.concatenate([d2[b] for b in range(K)], axis=1)
    d3_strip = np.concatenate([d3[b] for b in range(K)], axis=1)

    stages = [
        # name, matrix, cmap, 真实高度, 显示幅面权重, 描述
        ("input", inp_strip, "gray", 16, 1.0, f"(1, 16, 128) x{K}  |  2D input spectrogram"),
        ("conv2d_0+relu", c0_strip, "viridis", 64, 3.0, f"(16, 4, 128) -> 64 rows  |  height 16->4, 16ch x 4 rows"),
        ("conv2d_3+relu", c1_strip, "magma", 32, 2.0, f"(32, 1, 128)  |  height 4->1, compressed to 1D"),
        ("conv1d_1 dil=1", d1_strip, "magma", 64, 3.5, "(64, 128)  |  1D temporal, 32->64 ch"),
        ("conv1d_2 dil=2", d2_strip, "magma", 64, 3.5, "(64, 128)  |  dilated context x2"),
        ("conv1d_3 dil=4", d3_strip, "magma", 64, 3.5, "(64, 128)  |  dilated context x4"),
    ]

    n = len(stages)
    total_h = sum(s[4] for s in stages)
    fig, axes = plt.subplots(n, 1, figsize=(16, total_h * 1.8),
                             gridspec_kw={"height_ratios": [s[4] for s in stages],
                                          "hspace": 0.55})

    for ax, (name, mat, cmap, h, w, desc) in zip(axes, stages):
        v = np.percentile(mat, 99) + 1e-8
        ax.imshow(mat, aspect="auto", cmap=cmap, origin="lower",
                  vmin=0, vmax=v, extent=[0, W, 0, h])
        for b in range(1, K):
            ax.axvline(b * 128, color="cyan", lw=0.4, alpha=0.5)
        ax.set_ylabel(f"H={h}", fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"{name:16s}  {desc}", fontsize=9, loc="left",
                     family="monospace")

    axes[-1].set_xlabel(f"time (width = {K} x 128 = {W})")
    fig.suptitle("Feature Pipeline: 2D CNN (height 16->4->1) -> 1D CNN (temporal)",
                 fontsize=11, y=0.995)
    save_fig(fig, "feature_pipeline.png")


def main():
    device = torch.device("cpu")
    print(f"loading {CKPT}")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    vocab = ck.get("vocab", IDX2CHAR)
    model = CWModel(vocab_size=len(vocab))
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded: epoch={ck.get('epoch')} cer={ck.get('cer'):.4f}")

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 架构与参数概览 ----
    n_params = sum(p.numel() for p in model.parameters())
    summary = [
        f"CWModel v3  params={n_params/1e6:.3f}M  vocab={len(vocab)}",
        f"checkpoint epoch={ck.get('epoch')} cer={ck.get('cer'):.4f}",
        "",
        "== layers ==",
    ]
    for nm, m in model.named_modules():
        if nm == "":
            continue
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear,
                          torch.nn.GRU, torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            shp = tuple(m.weight.shape) if hasattr(m, "weight") else ""
            summary.append(f"{nm:18s} {type(m).__name__:12s} {shp}")
    (OUT / "model_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"  saved {(OUT / 'model_summary.txt').relative_to(ROOT)}")

    # ---- 所有卷积层卷积核 ----
    print("visualizing kernels:")
    viz_conv2d_first(model.cnn[0].weight.data, "conv2d_0")
    viz_conv2d_multi(model.cnn[3].weight.data, "conv2d_3")
    viz_conv1d(model.conv1d_1.weight.data, "conv1d_1")
    viz_conv1d(model.conv1d_2.weight.data, "conv1d_2")
    viz_conv1d(model.conv1d_3.weight.data, "conv1d_3")

    # ---- 取一条样本做特征图 ----
    with open(VAL_CSV) as f:
        rows = list(csv.DictReader(f))
    pick = None
    for r in rows:
        if int(r["noise_db"]) <= 0 and 30 <= int(r["num_blocks"]) <= 50 and 40 <= int(r["wpm"]) <= 50:
            pick = r
            break
    if pick is None:
        pick = rows[0]
    print(f"sample: text={pick['text']!r} wpm={pick['wpm']} noise={pick['noise_db']} "
          f"blocks={pick['num_blocks']}")

    arr = np.load(VAL_IMG / pick["path"]).astype(np.float32) / 255.0
    blocks = torch.from_numpy(arr).unsqueeze(1).unsqueeze(0)  # (1,K,1,16,128)
    num_blocks = torch.tensor([blocks.shape[1]], dtype=torch.long)

    K = blocks.shape[1]
    inp = blocks[0, :, 0].numpy()  # (K,16,128)
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.imshow(inp.reshape(K * 16, 128), aspect="auto", cmap="gray")
    ax.set_title(f"input: {K} blocks concatenated ({pick['text']})")
    ax.set_xlabel("freq bin")
    ax.set_ylabel("time (blocks stacked)")
    save_fig(fig, "feat_input.png")

    # ---- forward hooks 抓中间激活 ----
    acts: dict[str, torch.Tensor] = {}

    def mk_hook(key):
        def hook(_m, _i, o):
            if isinstance(o, torch.Tensor):
                acts[key] = o.detach()
            elif isinstance(o, tuple) and isinstance(o[0], torch.nn.utils.rnn.PackedSequence):
                # bigru 返回 (PackedSequence, h_n); 还原成 padded 序列
                packed = o[0]
                from torch.nn.utils.rnn import pad_packed_sequence
                padded, _ = pad_packed_sequence(packed, batch_first=True,
                                                total_length=K * CNN_T)
                acts[key] = padded.detach()
            else:
                acts[key] = o[0].detach()
        return hook

    handles = [
        model.cnn.register_forward_hook(mk_hook("cnn")),
        model.conv1d_1.register_forward_hook(mk_hook("conv1d_1")),
        model.conv1d_2.register_forward_hook(mk_hook("conv1d_2")),
        model.conv1d_3.register_forward_hook(mk_hook("conv1d_3")),
        model.bigru.register_forward_hook(mk_hook("bigru")),
        model.head.register_forward_hook(mk_hook("head")),
    ]

    with torch.no_grad():
        logits = model(blocks, num_blocks)
    for h in handles:
        h.remove()

    ilens = (num_blocks * CNN_T).long()

    feat_grid(acts["cnn"], "after CNN (2x conv2d+bn+relu)", "feat_after_cnn.png")
    feat_grid(acts["conv1d_1"], "after conv1d_1 (dil=1)+bn+relu",
              "feat_after_conv1d_1.png")
    feat_grid(acts["conv1d_2"], "after conv1d_2 (dil=2)+bn+relu",
              "feat_after_conv1d_2.png")
    feat_grid(acts["conv1d_3"], "after conv1d_3 (dil=4)+bn+relu",
              "feat_after_conv1d_3.png")
    feat_grid(acts["bigru"], "after BiGRU (3 layers, 256-dim)",
              "feat_after_bigru.png")
    feat_grid(acts["head"], "head logits over time", "feat_logits.png")

    # ---- 解码预览 (输入 + softmax + argmax 三行叠加, 按块对齐) ----
    pred = greedy_decode(logits, ilens, vocab)[0]
    L = int(ilens[0].item())  # K * CNN_T = K * 128
    K_blk = blocks.shape[1]
    UPS = CNN_T // 16  # 8, 每 block 输入 16 行时间 -> 128 解码步
    inp_up = np.repeat(blocks[0, :, 0].numpy(), UPS, axis=1)  # (K,128,128)
    inp_up = inp_up.reshape(-1, 128)[:L]  # (L,128)
    prob = torch.softmax(logits[0, :L], dim=-1).detach().cpu().numpy()
    argmax_path = prob.argmax(-1)

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 3, 1]})

    # 行1: 原始输入 spectrogram (时间上采样到 L 与解码对齐)
    axes[0].imshow(inp_up.T, aspect="auto", cmap="gray",
                   extent=[0, L, 0, 128], origin="lower")
    axes[0].set_ylabel("freq bin")
    axes[0].set_title(f"input spectrogram (upsampled x{UPS})  |  "
                      f"gt={pick['text']!r}  pred={pred!r}")

    # 行2: softmax 热力图
    im = axes[1].imshow(prob.T, aspect="auto", cmap="viridis",
                         extent=[0, L, 0, len(vocab)])
    axes[1].set_ylabel("class idx")
    axes[1].set_title("softmax(logits) over time")
    fig.colorbar(im, ax=axes[1], fraction=0.02, pad=0.01)

    # 行3: argmax 路径
    axes[2].plot(np.arange(L) + 0.5, argmax_path, lw=0.5, color="C0")
    axes[2].set_ylabel("argmax class")
    axes[2].set_xlabel("time step (K x 128, block-aligned)")
    axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))

    # 块边界竖线
    for ax in axes:
        for b in range(1, K_blk):
            ax.axvline(b * CNN_T, color="cyan", lw=0.4, alpha=0.4, zorder=5)
    axes[2].set_xlim(0, L)
    save_fig(fig, "decode_preview.png")

    # ---- conv1d_3 逐通道激活曲线 ----
    a = acts["conv1d_3"][0].detach().cpu().float()  # (BK,64,T)
    C = a.shape[1]
    aa = a.mean(0).numpy()  # (64, T) 跨 block 平均
    rows, cols = grid_dims(C)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 0.9))
    axes = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < C:
            ax.plot(aa[i], lw=0.6)
            ax.set_title(str(i), fontsize=6)
    fig.suptitle("conv1d_3 per-channel activation (mean over blocks)", fontsize=10)
    save_fig(fig, "feat_conv1d_3_perchannel.png")

    # ---- 特征流水线: 2D CNN 高度压缩 -> 1D CNN 时序特征 ----
    viz_feature_pipeline(model, vocab, pick)

    print("\ndone. outputs in", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
