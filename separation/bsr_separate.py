"""Command line interface for RoFormer-based vocal separation.

This utility loads a preset configuration and checkpoint to
isolate vocals from a stereo mixture. Output is a float32 WAV resampled
with FFmpeg's high-quality ``soxr`` backend (16 kHz mono by default)
 suitable for Parakeet ASR.
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
import yaml
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

PRESET_FILE = REPO_ROOT / "weights" / "roformer" / "presets.yaml"


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
    cfg = REPO_ROOT / info["cfg"]
    ckpt = REPO_ROOT / info["ckpt"]
    target = info.get("target", "vocals")
    return cfg, ckpt, target


def main() -> int:
    parser = argparse.ArgumentParser(description="RoFormer vocal isolation")
    parser.add_argument("--in_wav", required=True, help="Input stereo WAV")
    parser.add_argument("--out_wav", required=True, help="Output vocals WAV")
    parser.add_argument("--preset", default=None, help="Model preset name")
    parser.add_argument("--cfg", default=None, help="Path to model YAML")
    parser.add_argument("--ckpt", default=None, help="Path to model checkpoint")
    parser.add_argument("--engine", default="builtin")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--target", default=None, help="Override target stem")
    parser.add_argument("--final_sr", type=int, default=16000, help="Final output sample rate")
    parser.add_argument("--mono", dest="mono", action="store_true")
    parser.add_argument("--no-mono", dest="mono", action="store_false")
    parser.set_defaults(mono=True)
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
        # Expect 44.1k from Lua extraction; if not, let the model handle minor mismatch or
        # do a single high-quality hop *outside* Python to avoid multiple resamplings.

        out = sep.separate(audio, sr, target)

        # Write native-SR vocals to a temp WAV, then soxr→16k mono with FFmpeg (precision=28)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            tmp_wav = t.name
        sf.write(tmp_wav, out.astype(np.float32), sr, subtype="FLOAT")
        ff = ["ffmpeg", "-y", "-i", tmp_wav]
        if args.mono:
            ff += ["-ac", "1"]
        ff += [
            "-af", "aresample=resampler=soxr:precision=28",
            "-ar", str(args.final_sr),
            "-c:a", "pcm_f32le", args.out_wav,
        ]
        rc = subprocess.call(ff)
        Path(tmp_wav).unlink(missing_ok=True)
        if rc != 0:
            raise RuntimeError(f"ffmpeg soxr→{args.final_sr} failed with code {rc}")
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
