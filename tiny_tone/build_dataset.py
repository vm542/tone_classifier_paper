from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .textgrid import read_textgrid, Interval
from .features import (
    MelSpecConfig,
    load_audio_16k,
    compute_logmel,
    cmvn_per_utt,
    compute_pitch_track,
    pitch_stats,
    normalize_pitch,
    slice_and_resample_time,
    pitch_dct_and_voicing_for_segment,
)
from .io_parquet import ParquetRowWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TinyToneNet dataset (AISHELL-3 wav + TextGrid -> parquet)")
    p.add_argument("--wav-root", type=str, required=True, help="Root directory containing wavs (recursively)")
    p.add_argument("--textgrid-root", type=str, required=True, help="Root directory containing TextGrid files (recursively)")
    p.add_argument("--output", type=str, required=True, help="Output parquet path")
    p.add_argument("--tier", type=str, default="pinyins", help="TextGrid tier name (or substring) to use")
    p.add_argument("--include-tone5", type=int, default=0, choices=[0, 1])
    p.add_argument("--max-rows", type=int, default=0, help="0 = no limit")
    p.add_argument("--chunk-size", type=int, default=2048, help="Rows per parquet write chunk")
    return p.parse_args()


def tone_from_syllable_text(text: str) -> Optional[int]:
    t = text.strip()
    if not t:
        return None
    # common forms: "ni3", "ma5", sometimes with spaces
    last = t[-1]
    if last.isdigit():
        tone = int(last)
        if tone in (1, 2, 3, 4, 5):
            return tone
    return None


def tri_boundaries(intervals: List[Interval], idx: int) -> Tuple[float, float, int]:
    cur = intervals[idx]
    boundary_flag = 0

    if idx - 1 >= 0:
        prev = intervals[idx - 1]
        left = 0.5 * (prev.end + cur.start)
    else:
        left = cur.start
        boundary_flag = 1

    if idx + 1 < len(intervals):
        nxt = intervals[idx + 1]
        right = 0.5 * (cur.end + nxt.start)
    else:
        right = cur.end
        boundary_flag = 1

    # clamp
    left = max(0.0, float(left))
    right = max(left, float(right))
    return left, right, boundary_flag


def build_rows_for_utterance(
    wav_path: Path,
    tg_path: Path,
    tier_name: str,
    include_tone5: bool,
    cfg: MelSpecConfig,
    max_rows: int,
    rows_out: List[Dict],
) -> None:
    tg = read_textgrid(tg_path)
    tier = tg.get_interval_tier(tier_name)
    if tier is None:
        return

    # Keep only non-empty syllable texts
    sylls: List[Interval] = [it for it in tier.intervals if it.text.strip()]
    if not sylls:
        return

    y, sr = load_audio_16k(str(wav_path), target_sr=cfg.sr)
    logmel = compute_logmel(y, cfg)
    logmel = cmvn_per_utt(logmel)

    pitch = compute_pitch_track(y, sr=sr, hop_length=cfg.hop_length, fmin=cfg.f_min, fmax=cfg.f_max)
    pitch_mu, pitch_sigma = pitch_stats(pitch)
    pitch_norm_full = normalize_pitch(pitch, pitch_mu, pitch_sigma)

    # Convert pitch to shape [1, T] for slicing helper
    pitch_mat = pitch_norm_full.reshape(1, -1)

    # Speaker/utt ids: infer from path
    utterance_id = wav_path.stem
    speaker_id = wav_path.parent.name

    for i in range(len(sylls)):
        if max_rows and len(rows_out) >= max_rows:
            return

        tone = tone_from_syllable_text(sylls[i].text)
        if tone is None:
            continue
        if (tone == 5) and (not include_tone5):
            continue

        # current boundaries
        left_c, right_c, boundary_flag = tri_boundaries(sylls, i)

        # context segments (prev, cur, next)
        mel_ctx = np.zeros((3, cfg.n_mels, 64), dtype=np.float32)
        pitch_feats = []
        voi_feats = []

        for chan, j in enumerate([i - 1, i, i + 1]):
            if j < 0 or j >= len(sylls):
                # missing context -> zeros
                dct = np.zeros((12,), dtype=np.float32)
                voi = np.array([0.0, 64.0, 0.0, -1.0, -1.0], dtype=np.float32)
                pitch_feats.append(dct)
                voi_feats.append(voi)
                continue

            left, right, _bf = tri_boundaries(sylls, j)
            mel_seg = slice_and_resample_time(logmel, left, right, sr=sr, hop_length=cfg.hop_length, out_T=64)
            mel_ctx[chan] = mel_seg

            p_seg = slice_and_resample_time(pitch_mat, left, right, sr=sr, hop_length=cfg.hop_length, out_T=64)[0]
            dct, voi = pitch_dct_and_voicing_for_segment(p_seg, out_T=64, dct_k=12)
            pitch_feats.append(dct)
            voi_feats.append(voi)

        mel_ctx = mel_ctx.astype(np.float16)
        pitch_voicing = np.concatenate(pitch_feats + voi_feats, axis=0).astype(np.float32)

        # Aux based on current segment only
        dur_ms = (sylls[i].end - sylls[i].start) * 1000.0
        window_len_ms = (right_c - left_c) * 1000.0

        cur_voi = voi_feats[1]
        voiced_ratio = float(cur_voi[0])
        unvoiced_ratio = float(1.0 - voiced_ratio)
        first_voiced = cur_voi[3]
        last_voiced = cur_voi[4]
        if first_voiced >= 0 and last_voiced >= 0 and last_voiced >= first_voiced:
            voiced_span_ms = float((last_voiced - first_voiced + 1.0) * (cfg.hop_length / cfg.sr) * 1000.0)
        else:
            voiced_span_ms = 0.0

        aux = np.array(
            [
                float(np.log(max(dur_ms, 1.0))),
                float(pitch_mu),
                float(pitch_sigma),
                float(voiced_ratio),
                float(unvoiced_ratio),
                float(window_len_ms),
                float(voiced_span_ms),
                float(boundary_flag),
            ],
            dtype=np.float32,
        )

        rows_out.append(
            {
                "mel_ctx": mel_ctx,
                "pitch_voicing": pitch_voicing,
                "aux": aux,
                "label": int(tone),
                "speaker_id": speaker_id,
                "utterance_id": utterance_id,
                "syllable_index": int(i),
                "wav_path": str(wav_path),
                "textgrid_path": str(tg_path),
                "syllable_text": sylls[i].text,
            }
        )


def main() -> None:
    args = parse_args()
    wav_root = Path(args.wav_root)
    tg_root = Path(args.textgrid_root)

    wavs = {p.stem: p for p in wav_root.rglob("*.wav")}
    tg_iter = list(tg_root.rglob("*.TextGrid")) + list(tg_root.rglob("*.textgrid"))

    cfg = MelSpecConfig()
    rows: List[Dict] = []
    writer = ParquetRowWriter(args.output)
    total = 0

    for tg_path in tqdm(tg_iter, desc="TextGrids"):
        utt = tg_path.stem
        wav_path = wavs.get(utt)
        if wav_path is None:
            continue
        build_rows_for_utterance(
            wav_path=wav_path,
            tg_path=tg_path,
            tier_name=args.tier,
            include_tone5=bool(args.include_tone5),
            cfg=cfg,
            max_rows=int(args.max_rows),
            rows_out=rows,
        )
        # stream write
        if len(rows) >= int(args.chunk_size):
            writer.write_rows(rows)
            total += len(rows)
            rows.clear()

        if args.max_rows and (total + len(rows)) >= args.max_rows:
            break

    if rows:
        writer.write_rows(rows)
        total += len(rows)
        rows.clear()
    writer.close()

    print(f"Wrote {total} rows to {args.output}")


if __name__ == "__main__":
    main()
