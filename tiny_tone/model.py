from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


class DSConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: Tuple[int, int]):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.dw_act = nn.ReLU6(inplace=True)

        self.pw = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.pw_act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.dw_bn(x)
        x = self.dw_act(x)
        x = self.pw(x)
        x = self.pw_bn(x)
        x = self.pw_act(x)
        return x


class TinyToneNet150k(nn.Module):
    def __init__(self, num_classes: int = 4, use_pitch: bool = True, use_aux: bool = True):
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_pitch = bool(use_pitch)
        self.use_aux = bool(use_aux)

        # mel branch
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.ds1 = DSConv(32, 64, stride=(2, 2))
        self.ds2 = DSConv(64, 96, stride=(2, 2))
        self.ds3 = DSConv(96, 128, stride=(2, 2))
        self.ds4 = DSConv(128, 192, stride=(2, 2))
        self.ds5 = DSConv(192, 256, stride=(1, 2))
        self.pool = nn.AdaptiveAvgPool2d(1)

        # pitch branch
        if self.use_pitch:
            self.pitch_mlp = nn.Sequential(
                nn.Linear(51, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.10),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
            )
            pitch_dim = 32
        else:
            self.pitch_mlp = None
            pitch_dim = 0

        # aux branch
        if self.use_aux:
            self.aux_mlp = nn.Sequential(
                nn.Linear(8, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
            )
            aux_dim = 16
        else:
            self.aux_mlp = None
            aux_dim = 0

        fused_dim = 256 + pitch_dim + aux_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(128, self.num_classes),
        )

    def forward(
        self,
        mel_ctx: torch.Tensor,
        pitch_voicing: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return logits."""
        x = self.stem(mel_ctx)
        x = self.ds1(x)
        x = self.ds2(x)
        x = self.ds3(x)
        x = self.ds4(x)
        x = self.ds5(x)
        x = self.pool(x).flatten(1)

        feats = [x]

        if self.use_pitch:
            if pitch_voicing is None:
                raise ValueError("pitch_voicing is required when use_pitch=True")
            feats.append(self.pitch_mlp(pitch_voicing))

        if self.use_aux:
            if aux is None:
                raise ValueError("aux is required when use_aux=True")
            feats.append(self.aux_mlp(aux))

        z = torch.cat(feats, dim=1)
        logits = self.head(z)
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        mel_ctx: torch.Tensor,
        pitch_voicing: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        logits = self.forward(mel_ctx, pitch_voicing=pitch_voicing, aux=aux)
        if temperature is not None:
            logits = logits / float(temperature)
        return torch.softmax(logits, dim=-1)
