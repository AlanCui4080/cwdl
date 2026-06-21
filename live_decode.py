#!/usr/bin/env python3
"""Real-time Morse code decoder with GUI waterfall and live audio capture.

Pipeline: audio -> STFT spectrogram -> 128px blocks (hop 64px) ->
groups of 16 blocks (hop 8) -> modelv3 CNN+BiGRU -> CTC greedy decode.

Speed control: integer decimation (1x/2x/3x/4x).  Taking every Nth sample
halves the duration (doubles WPM) and doubles apparent frequency.  The
center frequency is implicitly multiplied by N to track the signal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import stft, windows

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QGroupBox,
    QSlider, QDoubleSpinBox, QComboBox, QFileDialog,
)

from modelv3 import CWModel, greedy_decode, CNN_T
from cnnsetv3 import IDX2CHAR

SAMPLE_RATE = 48000
N_FFT = 2048
MS_PER_PIXEL = 10.0
FREQ_SPAN_HZ = 375.0
BLOCK_W = 128
HOP_W = 64
GROUP_BLOCKS = 16
GROUP_HOP = 8
BUFFER_SEC = 60.0
PROCESS_INTERVAL_MS = 150
DISPLAY_FREQ_HZ = 1500.0

SEQ_SILENCE_THRESH = 3       # consecutive silent blocks to close a sequence
SEQ_MAX_BLOCKS = 48          # cap to avoid OOM
SEQ_ACTIVE_FRAC = 0.08       # fraction of active columns to call a block "active"

ROOT = Path(__file__).parent
DEFAULT_CKPT = ROOT / "checkpoints" / "best_v3.pt"

HOP_SAMPLES = int(round(SAMPLE_RATE * MS_PER_PIXEL / 1000.0))
BLOCK_HOP_SAMPLES = HOP_W * HOP_SAMPLES
HZ_PER_PIXEL = SAMPLE_RATE / N_FFT
MODEL_FREQ_PIXELS = int(round(FREQ_SPAN_HZ / HZ_PER_PIXEL))
DISP_FREQ_PIXELS = int(round(DISPLAY_FREQ_HZ / HZ_PER_PIXEL))


def _build_colormap(n=256):
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        t = i / (n - 1)
        if t < 0.25:
            cmap[i] = [0, 0, int(255 * t / 0.25)]
        elif t < 0.5:
            s = (t - 0.25) / 0.25
            cmap[i] = [0, int(255 * s), 255]
        elif t < 0.75:
            s = (t - 0.5) / 0.25
            cmap[i] = [int(255 * s), 255, int(255 * (1 - s))]
        else:
            s = (t - 0.75) / 0.25
            cmap[i] = [255, int(255 * (1 - s)), int(255 * s * 0.5)]
    return cmap


COLORMAP = _build_colormap()


class AudioCapture:
    """Ring-buffer audio capture with monotonic sample counter."""

    def __init__(self, sample_rate=SAMPLE_RATE, buffer_sec=BUFFER_SEC):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_sec)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0
        self.stream = None

    def callback(self, indata, frames, time, status):
        if status:
            print(f"Audio: {status}")
        data = indata[:, 0].copy()
        n = len(data)
        end = self.write_pos + n
        if end <= self.buffer_size:
            self.buffer[self.write_pos:end] = data
        else:
            first = self.buffer_size - self.write_pos
            self.buffer[self.write_pos:] = data[:first]
            self.buffer[:n - first] = data[first:]
        self.write_pos = (self.write_pos + n) % self.buffer_size
        self.total_written += n

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1,
            callback=self.callback, dtype=np.float32, latency="low")
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_buffer(self):
        """Return (chronological copy, abs_sample_of_first)."""
        if self.total_written < self.buffer_size:
            buf = self.buffer[:self.total_written].copy()
        elif self.write_pos == 0:
            buf = self.buffer.copy()
        else:
            buf = np.concatenate(
                [self.buffer[self.write_pos:], self.buffer[:self.write_pos]])
        start_abs = max(0, self.total_written - self.buffer_size)
        return buf, start_abs


def compute_spec(samples, center_freq):
    """Compute model spectrogram (16 freq bins) and display spectrogram."""
    if len(samples) < N_FFT:
        return (np.zeros((MODEL_FREQ_PIXELS, 0), dtype=np.uint8),
                np.zeros((DISP_FREQ_PIXELS, 0), dtype=np.uint8))

    hop = HOP_SAMPLES
    noverlap = N_FFT - hop
    win = windows.hamming(N_FFT, sym=False)
    _, _, Z = stft(samples, fs=SAMPLE_RATE, window=win,
                   nperseg=N_FFT, noverlap=noverlap, nfft=N_FFT,
                   boundary="zeros", padded=True)

    s_db = 20.0 * np.log10(np.abs(Z) + 1e-12)
    max_bin = s_db.shape[0]

    # Model band
    f_low = max(0.0, center_freq - FREQ_SPAN_HZ / 2.0)
    idx_low = int(round(f_low / HZ_PER_PIXEL))
    idx_high = idx_low + MODEL_FREQ_PIXELS
    if idx_high > max_bin:
        idx_high = max_bin
        idx_low = max(0, idx_high - MODEL_FREQ_PIXELS)
    model_band = s_db[idx_low:idx_high, :]
    if model_band.shape[0] < MODEL_FREQ_PIXELS:
        pad_h = MODEL_FREQ_PIXELS - model_band.shape[0]
        model_band = np.pad(model_band, ((0, pad_h), (0, 0)),
                            mode="constant", constant_values=model_band.min())
    lo, hi = float(model_band.min()), float(model_band.max())
    if hi <= lo:
        hi = lo + 1.0
    model_img = np.clip((model_band - lo) / (hi - lo) * 255, 0, 255)
    model_img = np.flipud(model_img).astype(np.uint8)

    # Display band
    disp_low = max(0, idx_low - (DISP_FREQ_PIXELS - MODEL_FREQ_PIXELS) // 2)
    disp_high = min(max_bin, disp_low + DISP_FREQ_PIXELS)
    disp_band = s_db[disp_low:disp_high, :]
    if disp_band.shape[0] < DISP_FREQ_PIXELS:
        pad_h = DISP_FREQ_PIXELS - disp_band.shape[0]
        disp_band = np.pad(disp_band, ((0, pad_h), (0, 0)),
                           mode="constant", constant_values=disp_band.min())
    lo, hi = float(disp_band.min()), float(disp_band.max())
    if hi <= lo:
        hi = lo + 1.0
    disp_img = np.clip((disp_band - lo) / (hi - lo) * 255, 0, 255)
    disp_img = np.flipud(disp_img).astype(np.uint8)

    return model_img, disp_img


def slice_blocks(img):
    """Slice spectrogram into (K, H, BLOCK_W) blocks with HOP_W overlap."""
    h, w = img.shape
    blocks = []
    start = 0
    while start + BLOCK_W <= w:
        blocks.append(img[:, start:start + BLOCK_W])
        start += HOP_W
    if not blocks:
        return np.zeros((0, h, BLOCK_W), dtype=img.dtype)
    return np.stack(blocks, axis=0)


def estimate_wpm(model_img, decimate=1):
    """Estimate WPM from spectrogram by detecting shortest on-burst.

    model_img: (16, W) uint8.  decimate: speed factor (2 = 2x faster).
    """
    if model_img.size == 0 or model_img.shape[1] < 10:
        return 0.0, 0.0
    energy = model_img.mean(axis=0).astype(np.float32)
    thr = energy.min() + 0.4 * (energy.max() - energy.min())
    on = energy > thr
    bursts = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            dur = j - i
            if dur >= 1:
                bursts.append(dur)
            i = j
        else:
            i += 1
    if len(bursts) < 2:
        return 0.0, 0.0
    bursts.sort()
    n = len(bursts)
    short_bursts = bursts[:max(1, n // 3)]
    dit_cols = float(np.median(short_bursts))
    if dit_cols < 1:
        dit_cols = 1.0
    # Each column = 10ms * decimate of real audio (decimation compresses time)
    dit_ms = dit_cols * MS_PER_PIXEL * decimate
    wpm = 1200.0 / dit_ms
    confidence = min(1.0, n / 10.0)
    return wpm, confidence


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    idx2char = ck.get("vocab", IDX2CHAR)
    model = CWModel(vocab_size=len(idx2char)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded {ckpt_path}  epoch={ck.get('epoch')}  cer={ck.get('cer')}")
    return model, idx2char


@torch.no_grad()
def decode_group(model, group_blocks, idx2char, device, commit_start_block=0):
    """Decode a group of blocks, committing output from commit_start_block."""
    K = group_blocks.shape[0]
    inp = torch.from_numpy(group_blocks).float().div_(255.0)
    inp = inp.unsqueeze(1).unsqueeze(0).to(device)
    nb = torch.tensor([K], dtype=torch.long)
    logits = model(inp, nb)
    start_t = commit_start_block * CNN_T
    total_t = K * CNN_T
    ilens = torch.tensor([total_t - start_t], dtype=torch.long)
    preds = greedy_decode(logits[:, start_t:, :], ilens, idx2char)
    return preds[0]


class WaterfallWidget(QLabel):
    """Scrolling waterfall spectrogram display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setStyleSheet("background-color: black;")
        self.wf_width = 800
        self.wf_height = 120
        self.wf_data = np.zeros((self.wf_height, self.wf_width), dtype=np.uint8)
        self._marker_frac = -1.0
        self._render()

    def set_marker(self, frac):
        self._marker_frac = max(-1.0, min(1.0, frac))
        self.update()

    def add_columns(self, new_data):
        if new_data.size == 0:
            return
        _, n_new = new_data.shape
        if new_data.shape[0] != self.wf_height:
            indices = np.linspace(0, new_data.shape[0] - 1, self.wf_height)
            resized = np.zeros((self.wf_height, n_new), dtype=np.uint8)
            for i, idx in enumerate(indices):
                lo = int(np.floor(idx))
                hi = min(int(np.ceil(idx)) + 1, new_data.shape[0])
                resized[i] = new_data[lo:hi, :].mean(axis=0).astype(np.uint8)
            new_data = resized
        if n_new >= self.wf_width:
            self.wf_data = new_data[:, -self.wf_width:].copy()
        else:
            self.wf_data = np.roll(self.wf_data, -n_new, axis=1)
            self.wf_data[:, -n_new:] = new_data
        self._render()
        self.update()

    def _render(self):
        h, w = self.wf_data.shape
        rgb = COLORMAP[self.wf_data]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._marker_frac < 0:
            return
        painter = QPainter(self)
        x = int(self._marker_frac * self.width())
        painter.setPen(QPen(QColor(255, 60, 60, 220), 2))
        painter.drawLine(x, 0, x, self.height())
        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()


class ModelInputWidget(QLabel):
    """Grayscale view of the model's actual input: 16 freq bins x K*128 time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setStyleSheet("background-color: black;")
        self._blocks = None
        self._commit_start = 0

    def update_input(self, group_blocks, commit_start=0):
        if group_blocks is None or group_blocks.shape[0] == 0:
            self.clear()
            return
        self._blocks = group_blocks
        self._commit_start = commit_start
        self._render()

    def _render(self):
        if self._blocks is None:
            return
        K, h, w = self._blocks.shape
        total_w = K * w
        combined = self._blocks.transpose(1, 0, 2).reshape(h, total_w)
        combined = np.ascontiguousarray(combined)
        qimg = QImage(combined.tobytes(), total_w, h, total_w,
                      QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(pixmap)
        sw = pixmap.width()
        sh = pixmap.height()
        px_per_col = sw / total_w
        painter.setPen(QPen(QColor(60, 60, 60, 180), 1))
        for b in range(1, K):
            x = int(b * w * px_per_col)
            painter.drawLine(x, 0, x, sh)
        if self._commit_start < K:
            painter.setPen(QPen(QColor(0, 255, 80, 200), 2))
            x1 = int(self._commit_start * w * px_per_col)
            x2 = int(K * w * px_per_col)
            painter.drawRect(x1, 0, x2 - x1 - 1, sh - 1)
        painter.end()
        self.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()


class MainWindow(QMainWindow):

    def __init__(self, ckpt_override=None):
        super().__init__()
        self.setWindowTitle("Morse Decoder - Live")
        self.setMinimumSize(900, 600)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.idx2char = IDX2CHAR
        self.audio = None
        self.running = False
        self.center_freq = 1000.0
        self.decimate = 1
        self._ckpt_override = ckpt_override

        self.prev_total_written = 0
        self.next_scan_abs = 0         # absolute pos of next block to scan
        self.seq_blocks = []           # accumulated blocks for current sequence
        self.silence_count = 0         # consecutive silent blocks
        self.seq_has_signal = False    # whether current sequence has signal
        self.preview_text = ""         # last preview decode (not yet committed)
        self.accumulated_text = ""

        self._build_ui()
        self._load_model()

        self.timer = QTimer()
        self.timer.timeout.connect(self._process)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(6)

        # --- Controls ---
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Center Freq (Hz):"))
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(200, 4000)
        self.freq_spin.setValue(1000)
        self.freq_spin.setSingleStep(10)
        ctrl.addWidget(self.freq_spin)

        ctrl.addWidget(QLabel("Gain:"))
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(1, 300)
        self.gain_slider.setValue(100)
        ctrl.addWidget(self.gain_slider)

        ctrl.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        for d in [1, 2, 3, 4]:
            self.speed_combo.addItem(f"{d}x", d)
        self.speed_combo.setToolTip(
            "Decimation factor: take every Nth sample.\n"
            "N=2 halves duration (doubles WPM), doubles apparent freq.\n"
            "Center freq is automatically multiplied by N.")
        ctrl.addWidget(self.speed_combo)

        self.btn_start = QPushButton("Start")
        self.btn_start.setCheckable(True)
        self.btn_start.clicked.connect(self._toggle)
        ctrl.addWidget(self.btn_start)

        ctrl.addWidget(QLabel("CKPT:"))
        ckpt_name = (Path(self._ckpt_override).name if self._ckpt_override
                     else (DEFAULT_CKPT.name if DEFAULT_CKPT.exists() else "N/A"))
        self.ckpt_label = QLabel(ckpt_name)
        self.ckpt_label.setStyleSheet("color: gray;")
        ctrl.addWidget(self.ckpt_label)

        btn_ckpt = QPushButton("...")
        btn_ckpt.setMaximumWidth(30)
        btn_ckpt.clicked.connect(self._select_ckpt)
        ctrl.addWidget(btn_ckpt)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # --- Waterfall ---
        wf_group = QGroupBox("Waterfall")
        wf_layout = QVBoxLayout(wf_group)
        self.waterfall = WaterfallWidget()
        wf_layout.addWidget(self.waterfall)
        self.freq_label = QLabel("")
        self.freq_label.setStyleSheet("color: gray; font-size: 10px;")
        wf_layout.addWidget(self.freq_label)
        layout.addWidget(wf_group, stretch=2)

        # --- Model input (grayscale) ---
        mi_group = QGroupBox("Model Input (16 x K*128, grayscale)")
        mi_layout = QVBoxLayout(mi_group)
        self.model_input = ModelInputWidget()
        mi_layout.addWidget(self.model_input)
        self.mi_label = QLabel("")
        self.mi_label.setStyleSheet("color: gray; font-size: 10px;")
        mi_layout.addWidget(self.mi_label)
        layout.addWidget(mi_group, stretch=1)

        # --- Decoded text ---
        text_group = QGroupBox("Decoded Text")
        text_layout = QVBoxLayout(text_group)
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFont(QFont("Monospace", 14))
        self.text_display.setStyleSheet(
            "background-color: #1a1a2e; color: #00ff88;")
        self.text_display.setMinimumHeight(80)
        text_layout.addWidget(self.text_display)

        btn_row = QHBoxLayout()
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_text)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        text_layout.addLayout(btn_row)
        layout.addWidget(text_group, stretch=1)

        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

    def _clear_text(self):
        self.accumulated_text = ""
        self.text_display.clear()

    def _reset_sequence(self):
        self.seq_blocks = []
        self.silence_count = 0
        self.seq_has_signal = False
        self.preview_text = ""

    def _run_decode(self, all_blocks, label_prefix):
        """Run model on all_blocks, update model_input and text display."""
        K = all_blocks.shape[0]
        self.model_input.update_input(all_blocks, 0)
        self.mi_label.setText(
            f"{label_prefix}: {K} blocks  |  {self.decimate}x dec")
        text = decode_group(self.model, all_blocks, self.idx2char,
                            self.device, 0)
        return text

    def _decode_sequence(self):
       """Final decode: all seq_blocks + 1 black block, commit to text."""
       if not self.seq_has_signal or not self.seq_blocks:
           self._reset_sequence()
           return
       black = np.zeros((MODEL_FREQ_PIXELS, BLOCK_W), dtype=np.uint8)
       all_blocks = np.stack(self.seq_blocks + [black], axis=0)
       text = self._run_decode(all_blocks, f"seq({len(self.seq_blocks)}+1blk)")
       if text:
            self.accumulated_text += text + "\n"
       self.text_display.setPlainText(self.accumulated_text)
       cursor = self.text_display.textCursor()
       cursor.movePosition(cursor.MoveOperation.End)
       self.text_display.setTextCursor(cursor)
       self._reset_sequence()

    def _preview_decode(self):
       """Preview decode: partial seq_blocks + 1 black block, no commit."""
       if not self.seq_blocks:
           return
       black = np.zeros((MODEL_FREQ_PIXELS, BLOCK_W), dtype=np.uint8)
       all_blocks = np.stack(self.seq_blocks + [black], axis=0)
       text = self._run_decode(all_blocks, f"preview({len(self.seq_blocks)}+1blk)")
       self.preview_text = text
       display = self.accumulated_text + text
       self.text_display.setPlainText(display)
       cursor = self.text_display.textCursor()
       cursor.movePosition(cursor.MoveOperation.End)
       self.text_display.setTextCursor(cursor)

    def _load_model(self):
        ckpt = Path(self._ckpt_override) if self._ckpt_override else DEFAULT_CKPT
        if ckpt.exists():
            try:
                self.model, self.idx2char = load_model(ckpt, self.device)
                self.status_label.setText(
                    f"Model: {ckpt.name}  device={self.device}")
            except Exception as e:
                self.status_label.setText(f"Model load failed: {e}")
        else:
            self.status_label.setText(f"Checkpoint not found: {ckpt}")

    def _select_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", str(ROOT / "checkpoints"),
            "PyTorch (*.pt *.pth);;All (*)")
        if path:
            try:
                self.model, self.idx2char = load_model(Path(path), self.device)
                self.ckpt_label.setText(Path(path).name)
                self.status_label.setText(f"Model: {Path(path).name}")
            except Exception as e:
                self.status_label.setText(f"Load failed: {e}")

    def _toggle(self):
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        self.center_freq = self.freq_spin.value()
        self.decimate = self.speed_combo.currentData()
        # Effective center freq after decimation (freq axis stretches by N)
        self.eff_freq = self.center_freq * self.decimate
        self.prev_total_written = 0
        self.next_scan_abs = 0
        self.seq_blocks = []
        self.silence_count = 0
        self.seq_has_signal = False
        self.preview_text = ""
        self.accumulated_text = ""
        self.text_display.clear()
        self.audio = AudioCapture()
        self.audio.start()
        self.timer.start(PROCESS_INTERVAL_MS)
        self.running = True
        self.btn_start.setText("Stop")
        self.btn_start.setChecked(True)
        self.freq_spin.setEnabled(False)
        self.speed_combo.setEnabled(False)
        self.waterfall.set_marker(-1)
        self.model_input.update_input(None)
        self.mi_label.setText("")
        self.status_label.setText(
            f"Capturing @ {self.center_freq:.0f} Hz  "
            f"eff {self.eff_freq:.0f} Hz  {self.decimate}x")

    def _stop(self):
        self.timer.stop()
        if self.audio:
            self.audio.stop()
        self.running = False
        self.btn_start.setText("Start")
        self.btn_start.setChecked(False)
        self.freq_spin.setEnabled(True)
        self.speed_combo.setEnabled(True)
        self.status_label.setText("Stopped")

    def _process(self):
        if self.model is None or self.audio is None:
            return

        samples, buf_start_abs = self.audio.get_buffer()
        total_written_orig = self.audio.total_written
        gain = self.gain_slider.value() / 100.0

        # Integer decimation: take every Nth sample.
        # This compresses time by N (WPM x N) and stretches freq axis by N.
        D = self.decimate
        if D > 1:
            samples = samples[::D]
            total_written = total_written_orig // D
            buf_start_abs = buf_start_abs // D
        else:
            total_written = total_written_orig

        samples = samples * gain

        if len(samples) < N_FFT:
            return

        try:
            model_img, disp_img = compute_spec(samples, self.eff_freq)
        except Exception:
            return

        # --- Waterfall: add only new columns ---
        new_samples = total_written - self.prev_total_written
        self.prev_total_written = total_written
        new_cols = max(0, new_samples // HOP_SAMPLES)
        if new_cols > 0 and disp_img.shape[1] > 0:
            cols_to_add = min(new_cols, disp_img.shape[1])
            self.waterfall.add_columns(disp_img[:, -cols_to_add:])

        # Frequency label
        lo_f = max(0.0, self.eff_freq - DISPLAY_FREQ_HZ / 2.0)
        hi_f = lo_f + DISPLAY_FREQ_HZ
        self.freq_label.setText(
            f"{lo_f:.0f} Hz - {hi_f:.0f} Hz  |  "
            f"center {self.eff_freq:.0f} Hz (input {self.center_freq:.0f} Hz)")

        # --- Sequential group decoding ---
        # --- Sequence detection and decoding ---
        blocks = slice_blocks(model_img)
        K = blocks.shape[0]
        if K == 0:
            return

        # Per-column energy and adaptive threshold for signal detection
        col_energy = model_img.mean(axis=0).astype(np.float32)
        thr = col_energy.min() + 0.4 * (col_energy.max() - col_energy.min())

        new_blocks_added = False

        while True:
            local_sample = self.next_scan_abs - buf_start_abs
            if local_sample < 0:
                self.next_scan_abs = buf_start_abs
                local_sample = 0
            bi = local_sample // BLOCK_HOP_SAMPLES
            if bi >= K:
                break

            block = blocks[bi]
            # Active if > SEQ_ACTIVE_FRAC of columns above threshold
            active_frac = float((block.mean(axis=0) > thr).mean())
            is_active = active_frac > SEQ_ACTIVE_FRAC

            if is_active:
                self.seq_blocks.append(block.copy())
                self.silence_count = 0
                self.seq_has_signal = True
                new_blocks_added = True
            else:
                if self.seq_has_signal:
                    self.silence_count += 1
                    if self.silence_count >= SEQ_SILENCE_THRESH:
                        self._decode_sequence()
                # else: leading silence, skip

            self.next_scan_abs += BLOCK_HOP_SAMPLES

        # Cap sequence length
        if len(self.seq_blocks) >= SEQ_MAX_BLOCKS:
            self._decode_sequence()

        # Preview decode when new blocks arrive and sequence is active
        if new_blocks_added and self.seq_has_signal and self.seq_blocks:
            self._preview_decode()

        # Waterfall marker
        if total_written > 0 and self.next_scan_abs > 0:
            samples_behind = total_written - self.next_scan_abs
            cols_behind = samples_behind / HOP_SAMPLES
            marker_frac = 1.0 - cols_behind / self.waterfall.wf_width
            self.waterfall.set_marker(marker_frac)
        else:
            self.waterfall.set_marker(-1)

        # WPM estimate
        wpm, conf = estimate_wpm(model_img, self.decimate)
        wpm_str = f"~{wpm:.0f} WPM" if conf > 0.1 else "WPM ?"
        range_str = ""
        if conf > 0.1:
            range_str = " (in range)" if 20 <= wpm <= 70 else " (out of range)"

        self.status_label.setText(
            f"Blocks: {K}  |  Text: {len(self.accumulated_text)} chars  "
            f"|  {wpm_str}{range_str}  |  {self.decimate}x  "
            f"|  {self.center_freq:.0f}->{self.eff_freq:.0f} Hz  "
            f"|  {self.device}")

    def closeEvent(self, event):
        self._stop()
        event.accept()


def main():
    ap = argparse.ArgumentParser(description="Real-time Morse decoder")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="checkpoint path (default: checkpoints/best_v3.pt)")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(ckpt_override=args.ckpt)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
