
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image
from scipy.signal import stft, windows

SAMPLE_RATE = 48000
N_FFT = 2048
MS_PER_PIXEL = 10.0
FREQ_SPAN_HZ = 375.0
TIME_PIXEL_UNIT = 64

def _derive_params(sample_rate: int = SAMPLE_RATE,
                   n_fft: int = N_FFT,
                   ms_per_pixel: float = MS_PER_PIXEL,
                   freq_span_hz: float = FREQ_SPAN_HZ):

    if n_fft < 2:
        raise ValueError("n_fft 太小")
    hop = int(round(sample_rate * ms_per_pixel / 1000.0))
    if hop < 1:
        raise ValueError("ms_per_pixel 太小或采样率太低，hop<1")
    noverlap = n_fft - hop
    if noverlap < 0:
        raise ValueError(
            f"hop({hop}) > n_fft({n_fft})，无法构成有效的 STFT，"
            "请减小 ms_per_pixel 或增大 n_fft"
        )

    hz_per_pixel = sample_rate / n_fft
    freq_pixels = int(round(freq_span_hz / hz_per_pixel))

    return {
        "n_fft": n_fft,
        "hop": hop,
        "noverlap": noverlap,
        "hz_per_pixel": hz_per_pixel,
        "freq_pixels": freq_pixels,
    }

def _to_mono_float64(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float64, copy=False)

def _normalize_to_uint8(s_db: np.ndarray,
                        db_floor: float | None,
                        db_range_db: float | None) -> np.ndarray:

    if db_floor is None:
        lo = float(s_db.min())
    else:
        lo = float(db_floor)

    if db_range_db is None:
        hi = float(s_db.max())
    else:
        hi = lo + float(db_range_db)

    if hi <= lo:
        hi = lo + 1.0

    out = (s_db - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    return (out * 255.0).astype(np.uint8)

def compute_spectrogram(samples: np.ndarray,
                        center_freq: float = 0.0,
                        sample_rate: int = SAMPLE_RATE,
                        n_fft: int = N_FFT,
                        ms_per_pixel: float = MS_PER_PIXEL,
                        freq_span_hz: float = FREQ_SPAN_HZ,
                        time_pixel_unit: int = TIME_PIXEL_UNIT,
                        db_floor: float | None = None,
                        db_range_db: float | None = None) -> np.ndarray:

    p = _derive_params(sample_rate, n_fft, ms_per_pixel, freq_span_hz)
    hz_per_pixel = p["hz_per_pixel"]
    hop = p["hop"]

    samples = _to_mono_float64(samples)

    win = windows.hamming(p["n_fft"], sym=False)
    f, _, Z = stft(
        samples,
        fs=sample_rate,
        window=win,
        nperseg=p["n_fft"],
        noverlap=p["noverlap"],
        nfft=p["n_fft"],
        boundary="zeros",
        padded=True,
    )

    n_frames_natural = samples.size // hop
    if Z.shape[1] > n_frames_natural:
        Z = Z[:, :n_frames_natural]

    s_db = 20.0 * np.log10(np.abs(Z) + 1e-12)

    if center_freq <= 0.0:
        f_low = 0.0
    else:
        f_low = center_freq - freq_span_hz / 2.0
    if f_low < 0.0:
        f_low = 0.0

    idx_low = int(round(f_low / hz_per_pixel))
    idx_high = idx_low + p["freq_pixels"]

    max_bin = s_db.shape[0]
    if idx_high > max_bin:
        idx_high = max_bin
        idx_low = max(0, idx_high - p["freq_pixels"])

    band = s_db[idx_low:idx_high, :]

    if band.shape[0] < p["freq_pixels"]:
        pad_h = p["freq_pixels"] - band.shape[0]
        band = np.pad(band, ((0, pad_h), (0, 0)),
                      mode="constant", constant_values=band.min())

    img = _normalize_to_uint8(band, db_floor, db_range_db)
    img = np.flipud(img)
    return img

def slice_blocks(img: np.ndarray,
    block_w: int = TIME_PIXEL_UNIT,
    hop_w: int | None = None) -> list[tuple[int, int, np.ndarray]]:

    if hop_w is None:
        hop_w = block_w // 2
    if hop_w < 1:
        raise ValueError("hop_w 必须 >= 1")

    h, w = img.shape
    pad_val = int(img.min())
    blocks: list[tuple[int, int, np.ndarray]] = []
    start = 0
    while True:
        end = start + block_w
        if end <= w:
            blocks.append((start, end, img[:, start:end]))
        else:
            tail = img[:, start:w]
            pad_w = block_w - tail.shape[1]
            block = np.pad(tail, ((0, 0), (0, pad_w)),
                           mode="constant", constant_values=pad_val)
            blocks.append((start, start + block_w, block))
            break
        if end == w:
            break
        start += hop_w
    return blocks

def generate_spectrogram_from_wav(wav_path: str | Path,
                                  center_freq: float = 0.0,
                                  start_sec: float = 0.0,
                                  **kwargs) -> np.ndarray:

    data, sr = sf.read(str(wav_path), dtype="float64")
    sample_rate = kwargs.pop("sample_rate", SAMPLE_RATE)
    if sr != sample_rate:
        raise ValueError(f"wav 采样率 {sr} 与配置 {sample_rate} 不一致")

    data = _to_mono_float64(data)
    start = int(round(start_sec * sr))
    if start > 0:
        data = data[start:]

    return compute_spectrogram(data,
                               center_freq=center_freq,
                               sample_rate=sample_rate,
                               **kwargs)

def save_image(arr: np.ndarray, out_path: str | Path) -> None:
    Image.fromarray(arr, mode="L").save(str(out_path))

def main():
    p = _derive_params()
    print(
        f"params: n_fft={p['n_fft']}  hop={p['hop']}  "
        f"noverlap={p['noverlap']}  hz/px={p['hz_per_pixel']:.4f}  "
        f"freq_pixels={p['freq_pixels']}  time_pixel_unit={TIME_PIXEL_UNIT}"
    )

    img = generate_spectrogram_from_wav(
        Path(__file__).parent / "experiment" / "sos_cw.wav",
        center_freq=1000.0,
    )
    out = Path(__file__).parent / "experiment" / "sos_spectrogram.png"
    save_image(img, out)
    print(f"shape={img.shape} dtype={img.dtype} -> {out}")

if __name__ == "__main__":
    main()
