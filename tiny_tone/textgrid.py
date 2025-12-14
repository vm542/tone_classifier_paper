from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import re


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class IntervalTier:
    name: str
    intervals: List[Interval]


@dataclass(frozen=True)
class TextGrid:
    tiers: List[IntervalTier]

    def get_interval_tier(self, name_hint: str) -> Optional[IntervalTier]:
        hint = name_hint.strip().lower()
        for t in self.tiers:
            if t.name.strip().lower() == hint:
                return t
        for t in self.tiers:
            if hint in t.name.strip().lower():
                return t
        return None


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def read_textgrid(path: str | Path) -> TextGrid:
    """Minimal Praat TextGrid reader for common AISHELL-3 style grids.

    Supports IntervalTier only.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    tiers: List[IntervalTier] = []
    i = 0

    def read_value(prefix: str) -> str:
        nonlocal i
        ln = lines[i]
        if "=" not in ln:
            raise ValueError(f"Expected '=' in line: {ln}")
        left, right = ln.split("=", 1)
        if left.strip() != prefix:
            raise ValueError(f"Expected '{prefix} =', got: {ln}")
        i += 1
        return right.strip()

    # Fast path: walk tiers by "item [k]" blocks
    while i < len(lines):
        ln = lines[i]
        # Praat format includes a wrapper line: "item []:" which is NOT a tier.
        # Only treat numbered blocks like "item [1]:" as tiers.
        if ln.startswith("item [") and re.match(r"^item\s+\[\d+\]:$", ln):
            i += 1
            # Expect: class, name, xmin, xmax, intervals
            tier_class = _strip_quotes(read_value("class"))
            tier_name = _strip_quotes(read_value("name"))
            _ = read_value("xmin")
            _ = read_value("xmax")
            if tier_class != "IntervalTier":
                # Skip unsupported tier types
                # Try to skip to next item
                while i < len(lines) and not lines[i].startswith("item ["):
                    i += 1
                continue
            n_intervals = int(read_value("intervals: size"))
            intervals: List[Interval] = []
            for _k in range(n_intervals):
                # intervals [k]:
                if not lines[i].startswith("intervals ["):
                    raise ValueError(f"Expected intervals [k] block, got: {lines[i]}")
                i += 1
                xmin = float(read_value("xmin"))
                xmax = float(read_value("xmax"))
                text = _strip_quotes(read_value("text"))
                intervals.append(Interval(start=xmin, end=xmax, text=text))
            tiers.append(IntervalTier(name=tier_name, intervals=intervals))
        elif ln.startswith("item []:"):
            i += 1
        else:
            i += 1

    return TextGrid(tiers=tiers)
