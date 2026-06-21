
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SAMPLE_RATE = 48000
MIN_SEGMENT_MS = 1280.0
GUARD_MS = 20.0
SEGMENT_ALIGN_MS = 10.0
TONE_FREQ = 1000.0
SHAPING_BW_HZ = 200.0
SHAPING_ORDER = 3
SHAPING_BW_JITTER_PCT = 0.20

WPM_JITTER_PCT = 0.20
SEQ_BIAS_PCT = 0.15
FM_RATE_HZ_RANGE = (0.2, 1.0)
FM_AMP_HZ_RANGE = (10.0, 100.0)

QRN_PROB = 0.85
QSB_PROB = 0.20

QRN_IMPULSE_PROB = 0.01
QRN_IMPULSE_GAIN = 60.0
QRN_BURST_SPARSITY = 0.99
QRN_BURST_GAIN_RANGE = (10.0, 200.0)
QRN_SPIKE_MS = (0.1, 3.0)
QRN_BURST_FRAC = (0.10, 0.45)
QRN_FILTER_BW_HZ = 500.0
QRN_FILTER_ORDER = 4
QRN_WHITE_PEAK_SIGMA = 4.0

QSB_BANDWIDTH = 0.05
QSB_DEPTH_MIX = ((0.10, 12.0), (0.20, 9.0), (0.70, 6.0))

MULTIPATH_PROB = 0.50
MULTIPATH_N_RANGE = (3, 5)
MULTIPATH_DELAY_UNITS = (0.1, 0.25)
MULTIPATH_GAIN_RANGE = (0.2, 0.6)

MORSE_MAP = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "'": ".----.", "!": "-.-.--",
    "/": "-..-.", "(": "-.--.", ")": "-.--.-", "&": ".-...", ":": "---...",
    ";": "-.-.-.", "=": "-...-", "+": ".-.-.", "-": "-....-", "_": "..--.-",
    "\"": ".-..-.", "$": "...-..-", "@": ".--.-.",
}

def _min_segment_samples() -> int:
    return int(round(SAMPLE_RATE * MIN_SEGMENT_MS / 1000.0))

def _align_samples() -> int:
    return int(round(SAMPLE_RATE * SEGMENT_ALIGN_MS / 1000.0))

def _guard_samples() -> int:
    return int(round(SAMPLE_RATE * GUARD_MS / 1000.0))

def _build_envelope(text: str, wpm: float
                    ) -> tuple[np.ndarray, list[tuple[str, int, int]]]:

    guard_n = _guard_samples()
    align_n = _align_samples()
    min_seg_n = _min_segment_samples()

    base_dot = 1.2 / wpm
    seq_bias = 1.0 + np.random.uniform(-SEQ_BIAS_PCT, SEQ_BIAS_PCT)
    base_dot *= seq_bias

    def jittered_units(units: float) -> int:
        factor = 1.0 + np.random.uniform(-WPM_JITTER_PCT, WPM_JITTER_PCT)
        dur = units * base_dot * factor
        return max(1, int(round(dur * SAMPLE_RATE)))

    pulses: list[tuple[int, int]] = []
    char_spans: list[tuple[str, int, int]] = []
    cursor = guard_n
    words = text.upper().split()
    for wi, word in enumerate(words):
        for ci, ch in enumerate(word):
            code = MORSE_MAP.get(ch)
            if code is None:
                continue
            ch_start = cursor
            for si, sym in enumerate(code):
                n = jittered_units(1.0 if sym == "." else 3.0)
                pulses.append((cursor, cursor + n))
                cursor += n
                if si < len(code) - 1:
                    cursor += jittered_units(1.0)
            char_spans.append((ch, ch_start, cursor))
            if ci < len(word) - 1:
                cursor += jittered_units(3.0)
        if wi < len(words) - 1:
            sp_start = cursor
            cursor += jittered_units(7.0)
            char_spans.append((" ", sp_start, cursor))

    code_end = cursor
    seg_n = max(min_seg_n, code_end + guard_n)
    if seg_n % align_n != 0:
        seg_n = ((seg_n + align_n - 1) // align_n) * align_n

    env = np.zeros(seg_n, dtype=np.float64)
    for a, b in pulses:
        env[a:b] = 1.0
    return env, char_spans

def _shape_envelope(env: np.ndarray) -> np.ndarray:

    factor = 1.0 + np.random.uniform(-SHAPING_BW_JITTER_PCT, SHAPING_BW_JITTER_PCT)
    bw = SHAPING_BW_HZ * factor
    sos = signal.butter(SHAPING_ORDER, bw,
                        btype="low", fs=SAMPLE_RATE, output="sos")
    return signal.sosfilt(sos, env)

def _triangle_freq_offset(n: int) -> np.ndarray:

    fm = np.random.uniform(*FM_RATE_HZ_RANGE)
    amp_abs = np.random.uniform(*FM_AMP_HZ_RANGE)
    sign = 1.0 if np.random.rand() < 0.5 else -1.0
    amp = sign * amp_abs
    phase0 = np.random.uniform(0.0, 2.0 * np.pi)

    t = np.arange(n) / SAMPLE_RATE
    tri = signal.sawtooth(2.0 * np.pi * fm * t + phase0, width=0.5)
    return amp * tri

def _add_noise(sig: np.ndarray, noise_db: float, sig_power: float) -> np.ndarray:

    noise_power = sig_power * (10.0 ** (noise_db / 10.0))
    noise = np.random.randn(sig.size) * np.sqrt(noise_power)
    return sig + noise

def _add_qrn(sig: np.ndarray, noise_amp: float) -> np.ndarray:

    out = sig.copy()
    n = out.size

    spikes = np.zeros(n, dtype=np.float64)

    impulse_mask = np.random.random(n) < QRN_IMPULSE_PROB
    impulses = (np.random.random(n) - 0.5)
    spikes[impulse_mask] += QRN_IMPULSE_GAIN * noise_amp * impulses[impulse_mask]

    lo_g, hi_g = QRN_BURST_GAIN_RANGE
    lo_ms, hi_ms = QRN_SPIKE_MS
    burst_len = min(n, int(round(np.random.uniform(*QRN_BURST_FRAC) * n)))
    if burst_len > 0:
        b_start = np.random.randint(0, n - burst_len + 1)
        mean_spike = SAMPLE_RATE * (lo_ms + hi_ms) / 2.0 / 1000.0
        n_slots = max(1, int(round(burst_len / mean_spike)))
        for _ in range(n_slots):
            if np.random.random() < QRN_BURST_SPARSITY:
                continue
            spike_ms = lo_ms * (hi_ms / lo_ms) ** np.random.random()
            spike_len = max(1, int(round(SAMPLE_RATE * spike_ms / 1000.0)))
            off = np.random.randint(0, burst_len)
            s0 = b_start + off
            s1 = min(n, s0 + spike_len)
            gain = lo_g * (hi_g / lo_g) ** np.random.random()
            spikes[s0:s1] += gain * noise_amp * (np.random.random(s1 - s0) - 0.5)

    f_lo = TONE_FREQ - QRN_FILTER_BW_HZ / 2.0
    f_hi = TONE_FREQ + QRN_FILTER_BW_HZ / 2.0
    sos = signal.butter(QRN_FILTER_ORDER, [f_lo, f_hi],
                        btype="band", fs=SAMPLE_RATE, output="sos")
    spikes = signal.sosfilt(sos, spikes)

    spike_peak = float(np.max(np.abs(spikes)))
    if spike_peak > 0:
        white_peak = QRN_WHITE_PEAK_SIGMA * noise_amp
        spikes *= white_peak / spike_peak

    return out + spikes

def _qsb_envelope(n: int, bandwidth: float = QSB_BANDWIDTH,
                  depth_db: float = 6.0) -> np.ndarray:

    navg = max(int(np.ceil(0.37 * SAMPLE_RATE / bandwidth)), 1)
    navg = min(navg, max(n // 2, 1))
    norm = np.sqrt(3.0 * navg)

    gen_n = max(n + 4 * navg, 2 * n)
    r = 2.0 * np.random.random(gen_n) - 1.0

    def _movavg(x: np.ndarray) -> np.ndarray:
        c = np.cumsum(x)
        out = np.empty_like(x)
        out[:navg] = c[:navg]
        out[navg:] = c[navg:] - c[:-navg]
        return out / navg

    g_full = np.abs(_movavg(_movavg(_movavg(r)))) * norm

    start = np.random.randint(0, gen_n - n + 1)
    g = g_full[start:start + n]

    gmax = float(g.max())
    if gmax <= 0.0:
        return np.ones(n)
    g = g / gmax
    floor = 10.0 ** (-depth_db / 20.0)
    g = floor + (1.0 - floor) * g
    return g

def _add_multipath(sig: np.ndarray, base_dot: float) -> np.ndarray:

    out = sig.copy()
    n = sig.size
    n_paths = np.random.randint(MULTIPATH_N_RANGE[0], MULTIPATH_N_RANGE[1] + 1)
    for _ in range(n_paths):
        delay_units = np.random.uniform(*MULTIPATH_DELAY_UNITS)
        delay = int(round(delay_units * base_dot * SAMPLE_RATE))
        if delay <= 0 or delay >= n:
            continue
        gain = np.random.uniform(*MULTIPATH_GAIN_RANGE)
        out[delay:] += gain * sig[:n - delay]
    return out

def compute_morse(text: str,
                  wpm: float = 40.0,
                  noise_db: float | None = None,
                  peak_normalize: bool = True,
                  qrn: bool | None = None,
                  qsb: bool | None = None,
                  multipath: bool | None = None,
                  qrn_noise_db: float | None = None,
                  return_spans: bool = False
                  ):

    if not 1.0 <= wpm <= 240.0:
        raise ValueError(f"wpm 超出合理范围: {wpm}")

    env, char_spans = _build_envelope(text, wpm)
    env_shaped = _shape_envelope(env)

    n = env_shaped.size

    f_offset = _triangle_freq_offset(n)
    inst_freq = TONE_FREQ + f_offset
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / SAMPLE_RATE
    carrier = np.sin(phase)

    sig = env_shaped * carrier

    if multipath is None:
        multipath = np.random.random() < MULTIPATH_PROB
    if multipath:
        sig = _add_multipath(sig, 1.2 / wpm)

    if qsb is None:
        qsb = np.random.random() < QSB_PROB
    if qsb:
        r = np.random.random()
        depth = QSB_DEPTH_MIX[-1][1]
        for frac, db in QSB_DEPTH_MIX:
            if r < frac:
                depth = db
                break
            r -= frac
        sig = sig * _qsb_envelope(n, depth_db=depth)

    if noise_db is not None:
        sig = _add_noise(sig, noise_db, 1.0)

    if qrn is None:
        qrn = np.random.random() < QRN_PROB
    if qrn:
        qrn_db = qrn_noise_db if qrn_noise_db is not None else (noise_db if noise_db is not None else 12.0)
        sig = _add_qrn(sig, np.sqrt(10.0 ** (qrn_db / 10.0)))

    if peak_normalize:
        peak = float(np.max(np.abs(sig)))
        if peak > 0:
            sig = sig / peak * 0.95

    sig = sig.astype(np.float32)
    if return_spans:
        return sig, char_spans
    return sig

def save_wav(samples: np.ndarray, path: str | Path) -> None:
    sf.write(str(path), samples, SAMPLE_RATE)

def main():
    out_dir = Path(__file__).parent / "experiment"
    out_dir.mkdir(exist_ok=True)

    samples = ["SOS", "PARIS", "CQ", "73", "QRZ", "12345", "HELLO"]
    for txt in samples:
        wpm = float(np.random.uniform(30.0, 60.0))
        sig = compute_morse(txt, wpm=wpm, noise_db=-5.0)
        out = out_dir / f"morse_{txt}.wav"
        save_wav(sig, out)
        print(f"{out.name}  wpm={wpm:.1f}  len={len(sig)} samples "
              f"({len(sig)/SAMPLE_RATE*1000:.1f} ms)")

if __name__ == "__main__":
    main()
