from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def require_pyarrow():
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Parquet support requires 'pyarrow'. Install it via: pip install pyarrow\n"
            f"Original error: {e}"
        )


def _rows_to_table(rows: List[Dict[str, Any]]):
    """Convert list-of-dicts rows to a pyarrow Table."""
    require_pyarrow()
    import pyarrow as pa

    if len(rows) == 0:
        raise ValueError("No rows to write")

    mel_flat = np.stack([r["mel_ctx"].reshape(-1).astype(np.float16, copy=False) for r in rows], axis=0)
    pitch = np.stack([r["pitch_voicing"].astype(np.float32, copy=False) for r in rows], axis=0)
    aux = np.stack([r["aux"].astype(np.float32, copy=False) for r in rows], axis=0)
    label = np.array([int(r["label"]) for r in rows], dtype=np.int16)

    speaker_id = [str(r.get("speaker_id", "")) for r in rows]
    utterance_id = [str(r.get("utterance_id", "")) for r in rows]
    syllable_index = np.array([int(r.get("syllable_index", -1)) for r in rows], dtype=np.int32)

    wav_path = [str(r.get("wav_path", "")) for r in rows]
    textgrid_path = [str(r.get("textgrid_path", "")) for r in rows]
    syllable_text = [str(r.get("syllable_text", "")) for r in rows]

    # fixed-size lists
    mel_arr = pa.FixedSizeListArray.from_arrays(
        pa.array(mel_flat.reshape(-1), type=pa.float16()),
        mel_flat.shape[1],
    )
    pitch_arr = pa.FixedSizeListArray.from_arrays(
        pa.array(pitch.reshape(-1), type=pa.float32()),
        pitch.shape[1],
    )
    aux_arr = pa.FixedSizeListArray.from_arrays(
        pa.array(aux.reshape(-1), type=pa.float32()),
        aux.shape[1],
    )

    return pa.table(
        {
            "mel_ctx": mel_arr,
            "pitch_voicing": pitch_arr,
            "aux": aux_arr,
            "label": pa.array(label, type=pa.int16()),
            "speaker_id": pa.array(speaker_id, type=pa.string()),
            "utterance_id": pa.array(utterance_id, type=pa.string()),
            "syllable_index": pa.array(syllable_index, type=pa.int32()),
            "wav_path": pa.array(wav_path, type=pa.string()),
            "textgrid_path": pa.array(textgrid_path, type=pa.string()),
            "syllable_text": pa.array(syllable_text, type=pa.string()),
        }
    )


def write_parquet_rows(output_path: str, rows: List[Dict[str, Any]]) -> None:
    """Write all rows to a single parquet file (in-memory)."""
    require_pyarrow()
    import pyarrow.parquet as pq

    table = _rows_to_table(rows)
    pq.write_table(table, output_path, compression="zstd")


class ParquetRowWriter:
    """Stream parquet writing in chunks to avoid holding the full dataset in memory."""

    def __init__(self, output_path: str, compression: str = "zstd"):
        require_pyarrow()
        import pyarrow.parquet as pq

        self.output_path = output_path
        self.compression = compression
        self._pq = pq
        self._writer = None

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        table = _rows_to_table(rows)
        if self._writer is None:
            self._writer = self._pq.ParquetWriter(self.output_path, table.schema, compression=self.compression)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _fixed_size_list_chunked_to_2d(col, item_size: int, dtype: np.dtype) -> np.ndarray:
    """Convert pyarrow (chunked) FixedSizeListArray column to a dense 2D ndarray.

    Arrow exposes FixedSizeListArray as a list column; `to_numpy()` yields an object array
    of per-row ndarrays, which is slow and error-prone. The `.values` buffer is flat and
    can be reshaped efficiently.
    """
    # col can be ChunkedArray from a table column
    try:
        arr = col.combine_chunks()
    except Exception:
        # Already an Array
        arr = col
    vals = arr.values.to_numpy(zero_copy_only=False).astype(dtype, copy=False)
    n = int(len(arr))
    if vals.size != n * item_size:
        raise ValueError(f"Unexpected flat size for fixed_size_list: got {vals.size}, expected {n * item_size}")
    return vals.reshape(n, item_size)


def read_parquet_arrays(path: str) -> Dict[str, Any]:
    """Load parquet into numpy arrays; returns dict with arrays and metadata."""
    require_pyarrow()
    import pyarrow.parquet as pq

    table = pq.read_table(path)

    # Arrays
    mel = _fixed_size_list_chunked_to_2d(table["mel_ctx"], 3 * 64 * 64, np.float16)
    pitch = _fixed_size_list_chunked_to_2d(table["pitch_voicing"], 51, np.float32)
    aux = _fixed_size_list_chunked_to_2d(table["aux"], 8, np.float32)

    label = np.asarray(table["label"].to_numpy(zero_copy_only=False)).astype(np.int64)
    speaker_id = table["speaker_id"].to_pylist()

    return {
        "mel_ctx": mel.reshape(-1, 3, 64, 64),
        "pitch_voicing": pitch,
        "aux": aux,
        "label": label,
        "speaker_id": np.array(speaker_id, dtype=object),
        "table": table,
    }
