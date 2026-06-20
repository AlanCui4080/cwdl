
import numpy as np
from PIL import Image
import soundfile as sf
from scipy.signal import stft, windows

def generate_spectrogram(
    wav_path,
    center_freq=0.0,
    n_fft_ms=100.0,
    overlap=0.707,
    freq_span=400.0,
):
    data, sr = sf.read(wav_path, dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)

    n_fft = int(round(sr * n_fft_ms / 1000.0))
    if n_fft < 2:
        raise ValueError("n_fft_ms too small for this sample rate")

    noverlap = int(round(n_fft * overlap))
    hop = n_fft - noverlap
    if hop < 1:
        hop = 1
        noverlap = n_fft - 1

    hz_per_px = sr / n_fft
    ms_per_px = hop / sr * 1000.0

    if center_freq == 0.0:
        f_low, f_high = 0.0, freq_span
    else:
        f_low = center_freq - freq_span / 2.0
        f_high = center_freq + freq_span / 2.0

    if f_low < 0:
        f_low = 0.0
    nyquist = sr / 2.0
    if f_high > nyquist:
        f_high = nyquist

    idx_low = int(round(f_low / hz_per_px))
    idx_high = int(round(f_high / hz_per_px))
    if idx_low < 0:
        idx_low = 0

    win = windows.hamming(n_fft, sym=False)
    f, t, Z = stft(
        data,
        fs=sr,
        window=win,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )

    S_db = 20 * np.log10(np.abs(Z) + 1e-12)

    n_freq_bins = idx_high - idx_low
    if n_freq_bins < 1:
        n_freq_bins = 1
    if idx_high > S_db.shape[0]:
        idx_high = S_db.shape[0]
    S_band = S_db[idx_low:idx_high, :]

    img_h, img_w = S_band.shape[0], S_band.shape[1]

    S_band = S_band - S_band.min()
    if S_band.max() > 0:
        S_band = S_band / S_band.max()
    img = (S_band * 255).astype(np.uint8)

    img = np.flipud(img)

    print(
        f"Spectrogram: {img_w}x{img_h}  "
        f"n_fft={n_fft} ({n_fft_ms:.1f} ms)  "
        f"overlap={overlap:.3f}  "
        f"hop={hop}  "
        f"{hz_per_px:.2f} Hz/px  "
        f"{ms_per_px:.2f} ms/px  "
        f"band={f_low:.0f}-{f_high:.0f} Hz"
    )
    return img

def main():
    img_arr = generate_spectrogram(
        "sos_cw.wav",
        center_freq=1000.0,
        n_fft_ms=25.0,
        overlap=0.804,
        freq_span=400.0,
    )
    img = Image.fromarray(img_arr, mode="L")
    img.save("sos_spectrogram.png")

if __name__ == "__main__":
    main()
