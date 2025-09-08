"""Command line interface for RoFormer-based vocal separation.

This utility loads a specific configuration and checkpoint to isolate
vocals from a stereo mixture. Output is a float32 WAV resampled with
FFmpeg's high-quality ``soxr`` backend (16 kHz mono by default) suitable
for Parakeet ASR.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa
torch.backends.cudnn.benchmark = True

# Allow execution without manipulating the environment. When launched
# directly (e.g., via `python separation/bsr_separate.py`), ensure the
# repository root is on ``sys.path`` so relative imports succeed.
REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))
    from separation.roformer_loader import load_separator  # type: ignore
else:
    from .roformer_loader import load_separator

def main() -> int:
    parser = argparse.ArgumentParser(description="RoFormer vocal isolation")
    parser.add_argument("--in_wav", required=True, help="Input stereo WAV")
    parser.add_argument("--out_wav", required=True, help="Output vocals WAV")
    parser.add_argument("--cfg", required=True, help="Path to model YAML")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--target", default="vocals", help="Target stem to extract")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda/cpu)")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_sr", type=int, help="Optional output sample rate (default: mixture SR)")
    parser.add_argument("--channels", type=int, choices=[1, 2], help="Optional output channels (default: mixture channels)")
    args = parser.parse_args()

    print(f"[SEP] loading cfg={args.cfg}")
    print(f"[SEP] loading ckpt={args.ckpt}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    sep = load_separator(str(args.cfg), str(args.ckpt), device=device, fp16=args.fp16)
    sep.model.to(device)
    sep.model.eval()

    try:
        audio, sr = sf.read(args.in_wav, dtype="float32")
        dur = len(audio) / float(sr)
        print(f"[SEP] input dur ~{dur:.1f}s @ {sr}Hz", file=sys.stderr)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)

        t0 = time.time()
        out = sep.separate(audio, sr, args.target)
        sep_secs = time.time() - t0
        print(f"[SEP] pure separation time: {sep_secs:.1f}s")
        if sep_secs < 10:
            raise RuntimeError("Separator finished too fast — likely fallback/no model.")

        out_sr = args.save_sr if args.save_sr else sr
        if out_sr != sr:
            out = librosa.resample(out.T, orig_sr=sr, target_sr=out_sr, axis=1).T
            sr = out_sr
        if args.channels is not None:
            if args.channels == 1 and out.ndim == 2:
                out = out.mean(axis=1, keepdims=True)
            elif args.channels == 2 and out.ndim == 1:
                out = np.repeat(out[:, None], 2, axis=1)
        sf.write(args.out_wav, out.astype(np.float32), sr, subtype="FLOAT")
    except Exception as e:
        print(f"Separation failed: {e}", file=sys.stderr)
        return 1
    finally:
        del sep
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
