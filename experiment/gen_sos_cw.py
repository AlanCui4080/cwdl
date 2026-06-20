import numpy as np
import soundfile as sf
from scipy import signal

WPM = 60
sr = 48000
tone_freq = 1000

def generate_morse(text, wpm=30, tone_freq=1000, sr=48000, fbw=200):
    dot_dur = 1.2 / wpm

    morse_map = {
        'S': '...', 'O': '---',
    }

    segments = []
    words = text.split()
    for wi, word in enumerate(words):
        for ci, ch in enumerate(word.upper()):
            code = morse_map.get(ch, '')
            for si, sym in enumerate(code):
                if sym == '.':
                    segments.append((1, dot_dur))
                else:
                    segments.append((1, 3 * dot_dur))
                if si < len(code) - 1:
                    segments.append((0, dot_dur))
            if ci < len(word) - 1:
                segments.append((0, 3 * dot_dur))
        if wi < len(words) - 1:
            segments.append((0, 7 * dot_dur))

    total_dur = sum(d for _, d in segments)
    total_samples = int(np.ceil(total_dur * sr))
    envelope = np.zeros(total_samples)
    t0 = 0
    for is_tone, dur in segments:
        n = int(round(dur * sr))
        if is_tone:
            envelope[t0:t0+n] = 1.0
        t0 += n

    if fbw > 0:
        sos = signal.butter(3, fbw, btype='low', fs=sr, output='sos')
        envelope = signal.sosfiltfilt(sos, envelope)

    t = np.arange(total_samples) / sr
    carrier = np.sin(2 * np.pi * tone_freq * t)
    signal_out = envelope * carrier

    return signal_out

sig = np.array([], dtype=np.float64)
target_len = int(1 * sr)

while len(sig) < target_len:
    chunk = generate_morse("SOS", WPM, tone_freq, sr, fbw=200)
    gap = np.zeros(int(2 * (1.2 / WPM) * sr))
    sig = np.concatenate([sig, chunk, gap])

sig = sig[:target_len]
sig = sig / np.max(np.abs(sig)) * 0.9

signal_power = np.mean(sig ** 2)
noise_std = np.sqrt(signal_power)
noise = noise_std * np.random.randn(len(sig))
signal_noisy = sig + noise

if np.max(np.abs(signal_noisy)) > 1.0:
    signal_noisy = signal_noisy / np.max(np.abs(signal_noisy)) * 0.99

sf.write("sos_cw.wav", signal_noisy.astype(np.float32), sr)
print(f"Generated sos_cw.wav: {len(signal_noisy)/sr:.2f}s, WPM={WPM}, sr={sr}, tone={tone_freq}Hz, SNR=0dB")
