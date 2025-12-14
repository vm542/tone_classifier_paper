import argparse
import json
import os
import random
import sys
from pathlib import Path

# Allow running as `python train/split_wavs.py` from repo root (or elsewhere)
if __package__ is None and __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

def get_spk_from_utt(utt: str):
    # Keep consistent with train/dataset/dataset.py, but avoid importing librosa/numba for simple splits.
    return utt[:7]

def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump_json(path: str, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Split utt2tones into train/val/test utterance lists.")
    parser.add_argument("--utt2tones", default="utt2tones.json", type=str)
    parser.add_argument("--out_dir", default="data", type=str)
    parser.add_argument("--seed", default=1024, type=int)

    # Speaker-based split sizes. Defaults preserve the original repo behavior for train size.
    parser.add_argument("--train_spks", default=174, type=int, help="Number of speakers to allocate to train")
    parser.add_argument("--val_spks", default=0, type=int, help="Number of speakers to allocate to validation")

    # Optional fixed test utterances list (e.g. paper test set). If provided, those utts are always test.
    parser.add_argument("--test_utts_json", default="test_utts.json", type=str)
    args = parser.parse_args()

    random.seed(args.seed, version=2)

    os.makedirs(args.out_dir, exist_ok=True)

    utt2tones: dict = _load_json(args.utt2tones)
    all_utts = list(utt2tones.keys())

    fixed_test = set()
    test_utts_path = Path(args.test_utts_json)
    if args.test_utts_json and test_utts_path.exists():
        test_list = _load_json(args.test_utts_json)
        fixed_test = set(test_list) & set(all_utts)
        print(f"Fixed test utts (intersection with utt2tones): {len(fixed_test)}")

    # speakers -> utts (excluding fixed test utts from the speaker split pool)
    spk2utts = {}
    for u in all_utts:
        if u in fixed_test:
            continue
        spk = get_spk_from_utt(u)
        spk2utts.setdefault(spk, []).append(u)

    speakers = list(spk2utts.keys())
    random.shuffle(speakers)
    print(f"Number of speakers (excluding fixed test utts): {len(speakers)}")

    train_spks = speakers[: args.train_spks]
    val_spks = speakers[args.train_spks : args.train_spks + args.val_spks]
    heldout_spks = speakers[args.train_spks + args.val_spks :]

    def get_utts_of_spks(spks: list):
        ret = []
        for spk in spks:
            ret += spk2utts[spk]
        return ret

    train_utts = get_utts_of_spks(train_spks)
    val_utts = get_utts_of_spks(val_spks)

    # test = fixed_test + remaining heldout speakers (if any)
    test_utts = sorted(fixed_test) + get_utts_of_spks(heldout_spks)

    print(f"Train size: {len(train_utts)} (spks={len(train_spks)})")
    print(f"Val size:   {len(val_utts)} (spks={len(val_spks)})")
    print(f"Test size:  {len(test_utts)} (fixed={len(fixed_test)}, heldout_spks={len(heldout_spks)})")

    _dump_json(f"{args.out_dir}/train_utts.json", train_utts)
    _dump_json(f"{args.out_dir}/val_utts.json", val_utts)
    _dump_json(f"{args.out_dir}/test_utts.json", test_utts)


if __name__ == "__main__":
    main()
