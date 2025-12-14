from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int = 2) -> float:
    y_true = y_true.astype(int)
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean(np.any(topk == y_true[:, None], axis=1)))


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for c in range(num_classes):
        m = (y_true == c)
        out[int(c)] = float(np.mean((y_pred[m] == c))) if np.any(m) else float("nan")
    return out


def multiclass_brier(probs: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
    y_true = y_true.astype(int)
    y_onehot = np.zeros((y_true.size, num_classes), dtype=np.float32)
    y_onehot[np.arange(y_true.size), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE on max-prob confidence."""
    y_true = y_true.astype(int)
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (conf >= b0) & (conf < b1)
        if not np.any(m):
            continue
        ece += float(np.abs(np.mean(acc[m]) - np.mean(conf[m])) * (np.sum(m) / conf.size))
    return float(ece)
