from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from cnnset import IDX2CHAR
from spectrogram import compute_spectrogram, _to_mono_float64

ROOT = Path(__file__).parent
BLOCK_W = 128
HOP_W = 64
GROUP_BLOCKS = 8

MODELS = {
    "v1": "model",
    "v1.1": "model_v11",
    "v2": "modelv2",
}


def detect_center_freq(samples, sr, fmin=200.0, fmax=2000.0):
    mono = _to_mono_float64(samples)
    n = len(mono)
    nfft = 1 << int(np.ceil(np.log2(max(n, 1))))
    xf = np.fft.rfftfreq(nfft, 1 / sr)
    mag = np.abs(np.fft.rfft(mono, n=nfft))
    mask = (xf >= fmin) & (xf <= fmax)
    if not mask.any():
        return 0.0
    return float(xf[mask][np.argmax(mag[mask])])


def slice_blocks(img, block_w=BLOCK_W, hop_w=HOP_W):
    h, w = img.shape
    pad_val = int(img.min())
    blocks, starts = [], []
    start = 0
    while True:
        end = start + block_w
        if end <= w:
            blocks.append(img[:, start:end])
            starts.append(start)
        else:
            tail = img[:, start:w]
            pad = block_w - tail.shape[1]
            blocks.append(
                np.pad(tail, ((0, 0), (0, pad)),
                       mode="constant", constant_values=pad_val))
            starts.append(start)
            break
        if end >= w:
            break
        start += hop_w
    return np.stack(blocks, axis=0), starts


def load_model(version, ckpt, device):
    import importlib
    mod = importlib.import_module(MODELS[version])
    ModelCls = getattr(mod, "CWModel")
    decode_fn = getattr(mod, "greedy_decode")
    cnn_t = getattr(mod, "CNN_T")

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    idx2char = ck.get("vocab", IDX2CHAR)
    model = ModelCls(vocab_size=len(idx2char)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {ckpt.name}  version={version}  "
          f"epoch={ck.get('epoch')}  cer={ck.get('cer')}")
    return model, decode_fn, cnn_t, idx2char


@torch.no_grad()
def decode_group(model, decode_fn, cnn_t, idx2char,
                 group_blocks, device):
    """解码一组 (<=GROUP_BLOCKS) 块, 与训练 forward 一致。"""
    k = group_blocks.shape[0]
    blocks = torch.from_numpy(group_blocks).float().unsqueeze(1).unsqueeze(0)
    num_blocks = torch.tensor([k], dtype=torch.long)
    blocks = blocks.to(device)
    num_blocks = num_blocks.to(device)
    logits = model(blocks, num_blocks)
    ilens = num_blocks * cnn_t
    preds = decode_fn(logits, ilens, idx2char)
    return preds[0]


def grouped_decode(model, decode_fn, cnn_t, idx2char,
                   all_blocks, device, group=GROUP_BLOCKS):
    """每 group 个块为一组送模型 (RNN 上下文组内拼合, 组间清空),
    与训练时整词序列送模型的方式一致。"""
    K = all_blocks.shape[0]
    out_chars = []
    for s in range(0, K, group):
        e = min(s + group, K)
        grp = all_blocks[s:e]
        pred = decode_group(model, decode_fn, cnn_t, idx2char, grp, device)
        out_chars.append(pred)
    return "".join(out_chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=str, help="wav 文件路径")
    ap.add_argument("--version", choices=list(MODELS), default="v1")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--center-freq", type=float, default=None)
    ap.add_argument("--group", type=int, default=GROUP_BLOCKS,
                    help="每多少块一组送 RNN (上下文组内拼合, 组间清空)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    wav = Path(args.wav)
    default_ckpt = {
        "v1": "checkpoints/best.pt",
        "v1.1": "checkpoints/best_v11.pt",
        "v2": "checkpoints/best_v2.pt",
    }[args.version]
    ckpt = Path(args.ckpt) if args.ckpt else ROOT / default_ckpt

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}")

    data, sr = sf.read(str(wav), dtype="float64")
    print(f"wav: {wav.name}  sr={sr}  samples={len(data)}  "
          f"dur={len(data)/sr:.2f}s")
    if sr != 48000:
        raise ValueError(f"采样率需 48000, 当前 {sr}")

    cf = args.center_freq if args.center_freq is not None \
        else detect_center_freq(data, sr)
    print(f"center_freq={cf:.1f} Hz")

    img = compute_spectrogram(data, center_freq=cf)
    print(f"spectrogram shape={img.shape}")

    blocks, starts = slice_blocks(img)
    all_blocks = blocks.astype(np.float32) / 255.0
    K = all_blocks.shape[0]
    print(f"blocks={K}  group={args.group}  "
          f"groups={int(np.ceil(K/args.group))}")

    model, decode_fn, cnn_t, idx2char = load_model(
        args.version, ckpt, device)

    pred = grouped_decode(model, decode_fn, cnn_t, idx2char,
                          all_blocks, device, group=args.group)

    print(f"\ndecoded ({len(pred)} chars):")
    print(pred)

    out_txt = wav.with_suffix(".decoded.txt")
    out_txt.write_text(pred, encoding="utf-8")
    print(f"\nsaved -> {out_txt}")

    spec_out = wav.with_suffix(".spectrogram.png")
    Image.fromarray(img, mode="L").save(str(spec_out))
    print(f"saved -> {spec_out}")


if __name__ == "__main__":
    main()
