import os
from pathlib import Path

from train.utils import onehot_encode

NUM_CLASSES = 6
EMBD_DIM = 128
SPEAKER_EMBEDDING_SIZE = 128
IN_PLANES = 16
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Configure via environment variables for local training.
# - Prefer `TONE_WAV_DIRS` (multiple roots, separated by ':' on macOS/Linux).
# - Fallback to `TONE_WAV_DIR` (single root).
# - If `data_aishell3/` exists in the project root, auto-use its train/test wav dirs.
# Example:
#   export TONE_WAV_DIRS=/path/to/data_aishell3/train/wav:/path/to/data_aishell3/test/wav
#   export TONE_WAV_DIR=/path/to/aishell3_wavs_all
_default_wav_dirs = [str(_PROJECT_ROOT / "wavs")]
_aishell3_root = _PROJECT_ROOT / "data_aishell3"
if _aishell3_root.exists():
    cand = []
    for p in [_aishell3_root / "train" / "wav", _aishell3_root / "test" / "wav"]:
        if p.exists():
            cand.append(str(p))
    if cand:
        _default_wav_dirs = cand

_wav_dirs_env = os.environ.get("TONE_WAV_DIRS")
if _wav_dirs_env:
    WAV_DIRS = [p for p in _wav_dirs_env.split(os.pathsep) if p]
else:
    _wav_dir_env = os.environ.get("TONE_WAV_DIR")
    WAV_DIRS = [_wav_dir_env] if _wav_dir_env else _default_wav_dirs

# Back-compat: keep a single WAV_DIR for callers that assume one root.
WAV_DIR = WAV_DIRS[0]
SPEAKER_EMBEDDING_DIR = os.environ.get("TONE_SPK_EMBD_DIR", str(_PROJECT_ROOT / "spk_embd"))
CACHE_DIR = 'exp/cache/'
MAX_GRAD_NORM = 1

PHONE_TO_ID = {
    'sil': 0,
    'a': 1,
    'ai': 2,
    'an': 3,
    'ang': 4,
    'ao': 5,
    'b': 6,
    'c': 7,
    'ch': 8,
    'd': 9,
    'e': 10,
    'ei': 11,
    'en': 12,
    'eng': 13,
    'er': 14,
    'f': 15,
    'g': 16,
    'h': 17,
    'i': 18,
    'ia': 19,
    'ian': 20,
    'iang': 21,
    'iao': 22,
    'ie': 23,
    'in': 24,
    'ing': 25,
    'iong': 26,
    'iu': 27,
    'j': 28,
    'k': 29,
    'l': 30,
    'm': 31,
    'n': 32,
    'o': 33,
    'ong': 34,
    'ou': 35,
    'p': 36,
    'q': 37,
    'r': 38,
    's': 39,
    'sh': 40,
    't': 41,
    'u': 42,
    'ua': 43,
    'uai': 44,
    'uan': 45,
    'uang': 46,
    'ui': 47,
    'un': 48,
    'uo': 49,
    'v': 50,
    'van': 51,
    've': 52,
    # 'ue': 52,
    "vn": 53,
    'w': 54,
    'x': 55,
    'y': 56,
    'z': 57,
    'zh': 58,
}

N_PHONES = len(list(set(list(PHONE_TO_ID.values()))))

PHONE_TO_ONEHOT = {
    p: onehot_encode(i, N_PHONES).astype('float32') for p, i in PHONE_TO_ID.items()
}
