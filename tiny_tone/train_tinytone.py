from __future__ import annotations

import argparse
import json
import os
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import trange

# macOS/conda setups sometimes crash on torch import with:
#   OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
# This opt-in workaround keeps the process alive. Override by setting TONE_KMP_DUPLICATE_LIB_OK=0.
if os.environ.get("TONE_KMP_DUPLICATE_LIB_OK", "1") == "1":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import confusion_matrix, log_loss

from .io_parquet import read_parquet_arrays
from .model import TinyToneNet150k
from .metrics import topk_accuracy, per_class_accuracy, multiclass_brier, expected_calibration_error
from .calibration import TemperatureScaler, softmax_np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TinyToneNet-150k")
    p.add_argument("--input", type=str, required=True, help="Parquet produced by tiny_tone/build_dataset.py")
    p.add_argument("--num-classes", type=int, default=4, choices=[4, 5])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--save-dir", type=str, default="tiny_tone_exp")
    p.add_argument("--max-rows", type=int, default=2000, help="Quick sanity run; 0=use all")

    p.add_argument("--cv", type=int, default=0, help="0 = single split; >0 = GroupKFold folds")
    p.add_argument("--calibrate", type=str, default="none", choices=["none", "temperature"])
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TinyToneParquetDataset(Dataset):
    def __init__(
        self,
        mel_ctx: np.ndarray,
        pitch: np.ndarray,
        aux: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        train: bool,
    ):
        self.mel_ctx = mel_ctx
        self.pitch = pitch
        self.aux = aux
        self.y = y
        self.w = sample_weight
        self.train = train

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        mel = self.mel_ctx[idx].astype(np.float32)

        if self.train:
            # random gain ±3dB on mel (additive in log domain)
            gain_db = np.random.uniform(-3.0, 3.0)
            mel = mel + (gain_db / 20.0) * np.log(10.0)

            # SpecAugment small (apply to all 3 channels)
            # freq mask up to 6 bins
            f = mel.shape[1]
            t = mel.shape[2]
            fmask = np.random.randint(0, 7)
            if fmask > 0:
                f0 = np.random.randint(0, max(1, f - fmask))
                mel[:, f0 : f0 + fmask, :] = 0.0
            # time mask up to 8 frames
            tmask = np.random.randint(0, 9)
            if tmask > 0:
                t0 = np.random.randint(0, max(1, t - tmask))
                mel[:, :, t0 : t0 + tmask] = 0.0

        x_mel = torch.from_numpy(mel)
        x_pitch = torch.from_numpy(self.pitch[idx].astype(np.float32))
        x_aux = torch.from_numpy(self.aux[idx].astype(np.float32))
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        w = torch.tensor(float(self.w[idx]), dtype=torch.float32)
        return x_mel, x_pitch, x_aux, y, w


def compute_sample_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    inv = inv / np.mean(inv)
    return inv[y]


def make_splits(y: np.ndarray, groups: np.ndarray, cv: int, seed: int):
    # If a small subset happens to contain too few speakers, fall back to sample-level splits.
    uniq_groups = np.unique(groups)
    if uniq_groups.size < 3 and (not cv or cv <= 1):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(y.shape[0])
        n = idx.size
        n_test = max(1, int(0.1 * n))
        n_dev = max(1, int(0.1 * (n - n_test)))
        te = idx[:n_test]
        dev = idx[n_test : n_test + n_dev]
        tr = idx[n_test + n_dev :]
        yield 0, tr, dev, te
        return

    if cv and cv > 1:
        gkf = GroupKFold(n_splits=cv)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(np.zeros_like(y), y, groups)):
            # split train into train/dev
            gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed + fold)
            (tr_idx, dev_idx) = next(gss.split(np.zeros_like(train_idx), y[train_idx], groups[train_idx]))
            tr = train_idx[tr_idx]
            dev = train_idx[dev_idx]
            yield fold, tr, dev, test_idx
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        train_idx, test_idx = next(gss.split(np.zeros_like(y), y, groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed + 1)
        tr_rel, dev_rel = next(gss2.split(np.zeros_like(train_idx), y[train_idx], groups[train_idx]))
        tr = train_idx[tr_rel]
        dev = train_idx[dev_rel]
        yield 0, tr, dev, test_idx


def cosine_warmup_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    # cosine decay
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def run_epoch(model, loader, device, optimizer=None, lr_sched=None, label_smoothing=0.0):
    train = optimizer is not None
    model.train(train)

    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")

    all_logits = []
    all_y = []

    total_loss = 0.0
    total_w = 0.0

    for step, (x_mel, x_pitch, x_aux, y, w) in enumerate(loader):
        # Be explicit about dtype: some environments set default dtype to float64.
        x_mel = x_mel.to(device=device, dtype=torch.float32)
        x_pitch = x_pitch.to(device=device, dtype=torch.float32)
        x_aux = x_aux.to(device=device, dtype=torch.float32)
        y = y.to(device)
        w = w.to(device)

        logits = model(x_mel, x_pitch, x_aux)
        loss_vec = ce(logits, y)
        loss = torch.sum(loss_vec * w) / torch.clamp(torch.sum(w), min=1e-6)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if lr_sched is not None:
                lr_sched(step)

        total_loss += float(loss.detach().cpu().item()) * float(torch.sum(w).detach().cpu().item())
        total_w += float(torch.sum(w).detach().cpu().item())

        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    y_np = np.concatenate(all_y, axis=0)
    avg_loss = total_loss / max(1e-6, total_w)
    return avg_loss, logits_np, y_np


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)

    data = read_parquet_arrays(args.input)
    mel = data["mel_ctx"].astype(np.float16)
    pitch = data["pitch_voicing"].astype(np.float32)
    aux = data["aux"].astype(np.float32)
    label_raw = data["label"].astype(np.int64)
    groups = data["speaker_id"]

    # Map labels 1..K -> 0..K-1
    y = label_raw - 1
    if args.num_classes == 4:
        m = (y >= 0) & (y < 4)
    else:
        m = (y >= 0) & (y < 5)

    mel, pitch, aux, y, groups = mel[m], pitch[m], aux[m], y[m], groups[m]

    if args.max_rows and mel.shape[0] > args.max_rows:
        # Random subset to avoid taking only one speaker/region from file ordering.
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(mel.shape[0], size=int(args.max_rows), replace=False)
        mel = mel[sel]
        pitch = pitch[sel]
        aux = aux[sel]
        y = y[sel]
        groups = groups[sel]

    results = {"args": vars(args), "folds": []}

    for fold, tr_idx, dev_idx, te_idx in make_splits(y, groups, args.cv, args.seed):
        model = TinyToneNet150k(num_classes=args.num_classes, use_pitch=True, use_aux=True).to(device).float()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        n_train_steps = int(math.ceil(len(tr_idx) / args.batch_size)) * args.epochs
        warmup_steps = int(args.warmup_frac * n_train_steps)
        global_step = {"v": 0}

        def lr_sched(_step_in_epoch: int):
            global_step["v"] += 1
            lr = cosine_warmup_lr(global_step["v"], n_train_steps, args.lr, warmup_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr

        w_all = compute_sample_weights(y[tr_idx], args.num_classes)

        ds_tr = TinyToneParquetDataset(mel[tr_idx], pitch[tr_idx], aux[tr_idx], y[tr_idx], w_all, train=True)
        ds_dev = TinyToneParquetDataset(mel[dev_idx], pitch[dev_idx], aux[dev_idx], y[dev_idx], np.ones(len(dev_idx)), train=False)
        ds_te = TinyToneParquetDataset(mel[te_idx], pitch[te_idx], aux[te_idx], y[te_idx], np.ones(len(te_idx)), train=False)

        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
        dl_dev = DataLoader(ds_dev, batch_size=args.batch_size, shuffle=False, num_workers=0)
        dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)

        best = {"dev_acc": -1.0, "state": None}

        for ep in trange(args.epochs, desc=f"fold{fold} epochs"):
            tr_loss, _, _ = run_epoch(model, dl_tr, device, optimizer=opt, lr_sched=lr_sched, label_smoothing=args.label_smoothing)
            dev_loss, dev_logits, dev_y = run_epoch(model, dl_dev, device)

            dev_probs = softmax_np(dev_logits)
            dev_pred = np.argmax(dev_probs, axis=1)
            dev_acc = float(np.mean(dev_pred == dev_y))

            if dev_acc > best["dev_acc"]:
                best["dev_acc"] = dev_acc
                best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        model.load_state_dict(best["state"])  # type: ignore[arg-type]

        # calibration on dev
        temperature = 1.0
        if args.calibrate == "temperature":
            _dev_loss, dev_logits, dev_y = run_epoch(model, dl_dev, device)
            ts = TemperatureScaler().fit(dev_logits, dev_y)
            temperature = ts.temperature

        # final test eval
        te_loss, te_logits, te_y = run_epoch(model, dl_te, device)
        te_probs = softmax_np(te_logits / float(temperature))
        te_pred = np.argmax(te_probs, axis=1)

        fold_res: Dict = {
            "fold": int(fold),
            "dev_best_acc": float(best["dev_acc"]),
            "temperature": float(temperature),
            "test": {},
        }

        fold_res["test"]["loss"] = float(te_loss)
        fold_res["test"]["acc"] = float(np.mean(te_pred == te_y))
        fold_res["test"]["top2_acc"] = float(topk_accuracy(te_probs, te_y, k=2))
        fold_res["test"]["per_class_acc"] = per_class_accuracy(te_y, te_pred, args.num_classes)
        fold_res["test"]["confusion_matrix"] = confusion_matrix(te_y, te_pred, labels=list(range(args.num_classes))).tolist()

        fold_res["test"]["log_loss"] = float(log_loss(te_y, te_probs, labels=list(range(args.num_classes))))
        fold_res["test"]["brier"] = float(multiclass_brier(te_probs, te_y, args.num_classes))
        fold_res["test"]["ece"] = float(expected_calibration_error(te_probs, te_y, n_bins=15))

        results["folds"].append(fold_res)

        # save checkpoint per fold
        ckpt_path = Path(args.save_dir) / f"tinytone_fold{fold}.pt"
        torch.save(
            {
                "model": "TinyToneNet150k",
                "num_classes": args.num_classes,
                "state_dict": model.state_dict(),
                "temperature": float(temperature),
            },
            ckpt_path,
        )

    out_path = Path(args.save_dir) / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
