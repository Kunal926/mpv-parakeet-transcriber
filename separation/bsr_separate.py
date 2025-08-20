"""Command line interface for RoFormer-based vocal separation.

This utility loads a preset configuration and checkpoint to
isolate vocals from a stereo mixture. Output is always a 16 kHz mono
WAV suitable for Parakeet ASR.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml

# Allow execution without manipulating the environment. When launched
# directly (e.g., via `python separation/bsr_separate.py`), ensure the
# repository root is on ``sys.path`` so relative imports succeed.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from separation.roformer_loader import load_separator  # type: ignore
else:
    from .roformer_loader import load_separator

PRESET_FILE = Path("weights/roformer/presets.yaml")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    """Resolve cfg, ckpt and target according to CLI/preset."""
    if args.cfg and args.ckpt:
        return Path(args.cfg), Path(args.ckpt), args.target or "vocals"

    with open(PRESET_FILE, "r", encoding="utf-8") as f:
        presets = yaml.safe_load(f)
    models = presets.get("models", {})
    preset_name = args.preset or presets.get("default")
    if preset_name not in models:
        raise ValueError(f"Unknown preset '{preset_name}'")
    info = models[preset_name]
    cfg = Path(info["cfg"])
    ckpt = Path(info["ckpt"])
    target = info.get("target", "vocals")
    return cfg, ckpt, target


def main() -> int:
    parser = argparse.ArgumentParser(description="RoFormer vocal isolation")
    parser.add_argument("--in_wav", required=True, help="Input stereo WAV (44.1kHz)")
    parser.add_argument("--out_wav", required=True, help="Output vocals WAV (16kHz mono)")
    parser.add_argument("--preset", default=None, help="Model preset name")
    parser.add_argument("--cfg", default=None, help="Path to model YAML")
    parser.add_argument("--ckpt", default=None, help="Path to model checkpoint")
    parser.add_argument("--engine", default="builtin")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--target", default=None, help="Override target stem")
    args = parser.parse_args()

    try:
        cfg_path, ckpt_path, target = resolve_paths(args)
    except Exception as e:
        print(f"Error resolving preset: {e}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading separator on {device} using preset '{args.preset or 'default'}'...", file=sys.stderr)
    sep = load_separator(str(cfg_path), str(ckpt_path), device=device, fp16=args.fp16)

    start = time.time()
    try:
        audio, sr = sf.read(args.in_wav, dtype="float32")
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)
        if sr != sep.sample_rate:
            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=sep.sample_rate, axis=1).T
            sr = sep.sample_rate

        out = sep.separate(audio, sr, target)

        # Downmix to mono and resample to 16 kHz
        mono = out.mean(axis=1)
        mono16 = librosa.resample(mono, orig_sr=sr, target_sr=16000)
        sf.write(args.out_wav, mono16.astype(np.float32), 16000, subtype="FLOAT")
    except Exception as e:
        print(f"Separation failed: {e}", file=sys.stderr)
        return 1
    finally:
        del sep
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elapsed = time.time() - start
    print(f"Separation complete in {elapsed:.2f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
