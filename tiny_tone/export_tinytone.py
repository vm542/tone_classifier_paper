from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import os

# macOS/conda setups sometimes crash on torch import with:
#   OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
# This opt-in workaround keeps the process alive. Override by setting TONE_KMP_DUPLICATE_LIB_OK=0.
if os.environ.get("TONE_KMP_DUPLICATE_LIB_OK", "1") == "1":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from .model import TinyToneNet150k


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export TinyToneNet-150k")
    p.add_argument("--checkpoint", type=str, required=True, help=".pt produced by tiny_tone/train_tinytone.py")
    p.add_argument("--out-dir", type=str, default="tiny_tone_export")
    p.add_argument("--onnx", type=int, default=1, choices=[0, 1])
    p.add_argument("--torchscript", type=int, default=1, choices=[0, 1])
    p.add_argument("--quantize-dynamic-int8", type=int, default=0, choices=[0, 1])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", 4))

    model = TinyToneNet150k(num_classes=num_classes, use_pitch=True, use_aux=True)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    if args.quantize_dynamic_int8:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # Example inputs
    mel = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    pitch = torch.zeros((1, 51), dtype=torch.float32)
    aux = torch.zeros((1, 8), dtype=torch.float32)

    if args.torchscript:
        traced = torch.jit.trace(model, (mel, pitch, aux))
        ts_path = out_dir / "tinytone150k.ts.pt"
        traced.save(str(ts_path))
        print(f"Saved TorchScript: {ts_path}")

    if args.onnx:
        onnx_path = out_dir / "tinytone150k.onnx"
        torch.onnx.export(
            model,
            (mel, pitch, aux),
            str(onnx_path),
            input_names=["mel_ctx", "pitch_voicing", "aux"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={
                "mel_ctx": {0: "B"},
                "pitch_voicing": {0: "B"},
                "aux": {0: "B"},
                "logits": {0: "B"},
            },
        )
        print(f"Saved ONNX: {onnx_path}")

    print("\nCoreML/TFLite notes:")
    print("- CoreML: use coremltools to convert the ONNX or TorchScript model (recommended: float16).")
    print("- TFLite: usually via ONNX -> TF -> TFLite, or re-export from a TF version.")
    print("- Post-training int8: prefer platform tooling; torch dynamic int8 only affects Linear layers.")


if __name__ == "__main__":
    main()
