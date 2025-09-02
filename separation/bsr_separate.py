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
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import tempfile

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
    parser.add_argument("--final_sr", type=int, default=16000, help="Final output sample rate")
    parser.add_argument("--mono", dest="mono", action="store_true")
    parser.add_argument("--no-mono", dest="mono", action="store_false")
    parser.set_defaults(mono=True)
    args = parser.parse_args()

    print(f"[SEP] loading cfg={args.cfg}")
    print(f"[SEP] loading ckpt={args.ckpt}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    sep = load_separator(str(args.cfg), str(args.ckpt), device=device, fp16=args.fp16)

    start = time.time()
    try:
        audio, sr = sf.read(args.in_wav, dtype="float32")
        dur = len(audio) / float(sr)
        print(f"[SEP] input dur ~{dur:.1f}s @ {sr}Hz", file=sys.stderr)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)
        total_samples = audio.shape[0]

        t0 = time.time()
        out = sep.separate(audio, sr, args.target)
        dur_s = total_samples / sr
        elapsed = time.time() - t0
        print(f"[SEP] processed ~{dur_s:.1f}s audio in {elapsed:.1f}s")
        if elapsed < 10:
            raise RuntimeError("Separator finished too fast — likely fallback/no model.")

        # Write native-SR vocals to a temp WAV, then soxr→16k mono with FFmpeg (precision=28)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            tmp_native = t.name
        sf.write(tmp_native, out.astype(np.float32), sr, subtype="FLOAT")
        cmd = [
            "ffmpeg", "-y", "-i", tmp_native, "-ac", "1",
            "-af", "aresample=resampler=soxr:precision=28", "-ar", str(args.final_sr),
            "-c:a", "pcm_f32le", args.out_wav,
        ]
        subprocess.run(cmd, check=True)
        Path(tmp_native).unlink(missing_ok=True)
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
