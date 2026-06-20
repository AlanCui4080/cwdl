
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

WPM_JITTER_PCT = 0.20
FM_RATE_HZ_RANGE = (0.2, 1.0)
FM_AMP_HZ_RANGE = (10.0, 100.0)

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
            cursor += jittered_units(7.0)

    code_end = cursor
    seg_n = max(min_seg_n, code_end + guard_n)
    if seg_n % align_n != 0:
        seg_n = ((seg_n + align_n - 1) // align_n) * align_n

    env = np.zeros(seg_n, dtype=np.float64)
    for a, b in pulses:
        env[a:b] = 1.0
    return env, char_spans

def _shape_envelope(env: np.ndarray) -> np.ndarray:

    sos = signal.butter(SHAPING_ORDER, SHAPING_BW_HZ,
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

def compute_morse(text: str,
                  wpm: float = 40.0,
                  noise_db: float | None = None,
                  peak_normalize: bool = True,
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

    noise = noise_db if noise_db is not None else -30.0
    sig = _add_noise(sig, noise, 1.0)

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
