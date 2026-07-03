"""Offline mel-feature extraction for Speech Commands KWS (numpy + stdlib only)."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
FRAME_LENGTH = 320
FRAME_STEP = 160
N_FFT = 512
NUM_MEL_BINS = 10
FEATURE_H = 16
FEATURE_W = 10
FEATURE_C = 1


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(num_bins: int, n_fft: int, sample_rate: int) -> np.ndarray:
    low_hz = 40.0
    high_hz = min(4000.0, sample_rate / 2.0)
    low_mel = _hz_to_mel(low_hz)
    high_mel = _hz_to_mel(high_hz)
    mel_points = np.linspace(low_mel, high_mel, num_bins + 2, dtype=np.float64)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points], dtype=np.float64)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int32)
    filters = np.zeros((num_bins, n_fft // 2 + 1), dtype=np.float32)
    for i in range(num_bins):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left or right == center:
            continue
        for k in range(left, center):
            if 0 <= k < filters.shape[1]:
                filters[i, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            if 0 <= k < filters.shape[1]:
                filters[i, k] = (right - k) / max(right - center, 1)
    return filters


_MEL_FB = _mel_filterbank(NUM_MEL_BINS, N_FFT, SAMPLE_RATE)


def read_wav_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM in {path}")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        # Simple linear resample for offline export scripts.
        x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=int(audio.shape[0] * SAMPLE_RATE / rate), endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio / 32768.0


def waveform_to_feature_map(audio: np.ndarray) -> np.ndarray:
    """Return 16x10x1 float feature map from a mono waveform."""
    if audio.size == 0:
        return np.zeros((FEATURE_H, FEATURE_W, FEATURE_C), dtype=np.float32)

    window = np.hanning(FRAME_LENGTH).astype(np.float32)
    num_frames = 1 + max(0, (audio.shape[0] - FRAME_LENGTH) // FRAME_STEP)
    mel_frames: list[np.ndarray] = []
    for frame_idx in range(num_frames):
        start = frame_idx * FRAME_STEP
        chunk = audio[start : start + FRAME_LENGTH]
        if chunk.shape[0] < FRAME_LENGTH:
            chunk = np.pad(chunk, (0, FRAME_LENGTH - chunk.shape[0]))
        spectrum = np.fft.rfft(chunk * window, n=N_FFT)
        power = (np.abs(spectrum) ** 2).astype(np.float32)
        mel = _MEL_FB @ power
        mel = np.log(mel + 1e-6)
        mel_frames.append(mel.astype(np.float32))

    if not mel_frames:
        mel_frames = [np.full(NUM_MEL_BINS, -10.0, dtype=np.float32)]

    mel_spec = np.stack(mel_frames, axis=0)  # [time, mel]
    # Resize time axis to FEATURE_H.
    src_t = mel_spec.shape[0]
    dst = np.zeros((FEATURE_H, FEATURE_W), dtype=np.float32)
    for t_out in range(FEATURE_H):
        src_pos = t_out * (src_t - 1) / max(FEATURE_H - 1, 1)
        left = int(math.floor(src_pos))
        right = min(left + 1, src_t - 1)
        alpha = src_pos - left
        frame = (1.0 - alpha) * mel_spec[left] + alpha * mel_spec[right]
        dst[t_out, :] = frame

    dst -= dst.min()
    peak = dst.max()
    if peak > 0.0:
        dst /= peak
    return dst.reshape(FEATURE_H, FEATURE_W, FEATURE_C)
