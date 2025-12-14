from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray, max_iter: int = 200) -> "TemperatureScaler":
        """Fit temperature on dev set logits by minimizing NLL (no leakage if dev is separate)."""
        device = torch.device("cpu")
        logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
        y_t = torch.tensor(y.astype(np.int64), dtype=torch.long, device=device)

        t = torch.nn.Parameter(torch.ones((), device=device))

        optim = torch.optim.LBFGS([t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
        ce = torch.nn.CrossEntropyLoss()

        def closure():
            optim.zero_grad(set_to_none=True)
            loss = ce(logits_t / torch.clamp(t, min=1e-3), y_t)
            loss.backward()
            return loss

        optim.step(closure)
        self.temperature = float(torch.clamp(t.detach(), min=1e-3).cpu().item())
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return logits / float(self.temperature)


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)
