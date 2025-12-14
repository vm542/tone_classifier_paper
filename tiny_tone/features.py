from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Some environments can crash on librosa import due to numba caching issues
# (\"no locator available\" when cache=True decorators are evaluated).
# Hard-disable numba JIT to make librosa import reliable for dataset building.
# This is slower but robust, and only affects feature extraction.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")

import librosa


@dataclass(frozen=True)
class MelSpecConfig:
    sr: int = 16000
    n_fft: int = 256
    win_length: int = 208
    hop_length: int = 80
    n_mels: int = 64
    f_min: float = 50.0
    f_max: float = 350.0
    eps: float = 1e-6


def load_audio_16k(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    return y, target_sr


def compute_logmel(y: np.ndarray, cfg: MelSpecConfig) -> np.ndarray:
    # mel power spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window="hann",
        center=False,
        power=2.0,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
    )
    logmel = np.log(mel + cfg.eps)
    return logmel.astype(np.float32)


def cmvn_per_utt(logmel: np.ndarray) -> np.ndarray:
    mu = float(np.mean(logmel))
    sigma = float(np.std(logmel) + 1e-8)
    return ((logmel - mu) / sigma).astype(np.float32)


def time_to_frame(t_sec: float, sr: int, hop_length: int) -> int:
    # center=False -> frame i starts at sample i*hop
    return int(np.floor(t_sec * sr / hop_length))


def slice_and_resample_time(m: np.ndarray, start_t: float, end_t: float, *, sr: int, hop_length: int, out_T: int = 64) -> np.ndarray:
    """Slice mel/pitch-like matrix [F, T] by time in seconds and resample to out_T frames."""
    if end_t <= start_t:
        return np.zeros((m.shape[0], out_T), dtype=np.float32)

    start_f = max(0, time_to_frame(start_t, sr, hop_length))
    end_f = max(start_f + 1, time_to_frame(end_t, sr, hop_length))

    T = m.shape[1]
    start_f = min(start_f, T - 1)
    end_f = min(end_f, T)

    seg = m[:, start_f:end_f]
    if seg.shape[1] <= 0:
        return np.zeros((m.shape[0], out_T), dtype=np.float32)

    if seg.shape[1] == out_T:
        return seg.astype(np.float32)

    x_old = np.linspace(0.0, 1.0, num=seg.shape[1], endpoint=True, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=out_T, endpoint=True, dtype=np.float32)

    out = np.empty((seg.shape[0], out_T), dtype=np.float32)
    for i in range(seg.shape[0]):
        out[i] = np.interp(x_new, x_old, seg[i].astype(np.float32))
    return out


def compute_pitch_track(y: np.ndarray, *, sr: int, hop_length: int, fmin: float = 50.0, fmax: float = 350.0) -> np.ndarray:
    """Return pitch per frame in Hz with np.nan for unvoiced.

    Uses librosa.yin (fast, no extra deps). Frame length is chosen to be stable.
    """
    frame_length = 1024
    pitch = librosa.yin(y=y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length, frame_length=frame_length)
    # librosa.yin returns f0 for all frames; mark very low confidence as nan is not available.
    # We treat values outside [fmin,fmax] as unvoiced.
    pitch = pitch.astype(np.float32)
    pitch[(pitch < fmin) | (pitch > fmax)] = np.nan
    return pitch


def pitch_stats(pitch: np.ndarray) -> Tuple[float, float]:
    voiced = pitch[np.isfinite(pitch)]
    if voiced.size == 0:
        return 0.0, 1.0
    mu = float(np.mean(voiced))
    sigma = float(np.std(voiced) + 1e-6)
    return mu, sigma


def normalize_pitch(pitch: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    p = pitch.copy().astype(np.float32)
    voiced = np.isfinite(p)
    p[voiced] = (p[voiced] - mu) / sigma
    p[~voiced] = 0.0
    return p


def _dct2_first_k(x: np.ndarray, k: int = 12) -> np.ndarray:
    """Compute first-k DCT-II coefficients (non-orthonormal) without scipy."""
    x = x.astype(np.float32)
    N = x.size
    if N == 0:
        return np.zeros((k,), dtype=np.float32)
    n = np.arange(N, dtype=np.float32)
    coeffs = np.empty((k,), dtype=np.float32)
    for ii in range(k):
        coeffs[ii] = float(np.sum(x * np.cos(np.pi * (n + 0.5) * ii / N)))
    return coeffs


def voicing_summary_from_pitch_norm(pitch_norm: np.ndarray) -> np.ndarray:
    """pitch_norm is 1D, with 0 for unvoiced."""
    x = pitch_norm
    voiced = x != 0.0
    n = x.size
    if n == 0:
        return np.array([0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)

    voiced_ratio = float(np.mean(voiced))

    # longest unvoiced run
    longest = 0
    cur = 0
    for v in voiced:
        if not v:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    # transitions voiced/unvoiced
    transitions = int(np.sum(voiced[1:] != voiced[:-1])) if n >= 2 else 0

    idx = np.where(voiced)[0]
    first = float(idx[0]) if idx.size else -1.0
    last = float(idx[-1]) if idx.size else -1.0

    return np.array([voiced_ratio, float(longest), float(transitions), first, last], dtype=np.float32)


def pitch_dct_and_voicing_for_segment(pitch_norm: np.ndarray, out_T: int = 64, dct_k: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Resample pitch_norm to out_T, then compute DCT + voicing summary."""
    if pitch_norm.size == 0:
        p = np.zeros((out_T,), dtype=np.float32)
    else:
        x_old = np.linspace(0.0, 1.0, num=pitch_norm.size, endpoint=True, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=out_T, endpoint=True, dtype=np.float32)
        p = np.interp(x_new, x_old, pitch_norm.astype(np.float32)).astype(np.float32)

    dct = _dct2_first_k(p, k=dct_k)
    voi = voicing_summary_from_pitch_norm(p)
    return dct, voi
