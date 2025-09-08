"""Command line interface for mpv-parakeet."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from . import ffmpeg_io, transcribe

__all__ = ["main", "transcribe"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="parakeet-transcribe")
    p.add_argument("audio", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--stdout", action="store_true")
    p.add_argument("--audio-start-offset", type=float, default=0.0)
    p.add_argument("--precision", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk-secs", type=float, default=30.0)
    p.add_argument("--overlap-secs", type=float, default=5.0)
    p.add_argument("--language", default="en")
    p.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v2")
    p.add_argument("--segmenter", choices=["word", "segment"], default="word")
    p.add_argument("--max-words", type=int, default=12)
    p.add_argument("--max-duration", type=float, default=6.0)
    p.add_argument("--pause", type=float, default=0.6)
    p.add_argument("--ffmpeg-path", default="ffmpeg")
    p.add_argument("--ffprobe-path", default="ffprobe")
    p.add_argument("--ffmpeg-filters")
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.output is None and not args.stdout:
        parser.error("--output or --stdout required")

    tmp_wav = Path(str(args.audio) + ".wav")
    ffmpeg_io.extract_audio(args.audio, tmp_wav, ffmpeg=args.ffmpeg_path, filters=args.ffmpeg_filters)
    start = ffmpeg_io.probe_start(args.audio, ffprobe=args.ffprobe_path)

    cfg = transcribe.TranscribeConfig(
        audio_path=tmp_wav,
        output_path=args.output,
        audio_start_offset=start + args.audio_start_offset,
        precision=args.precision,
        device=args.device,
        batch_size=args.batch_size,
        chunk_secs=args.chunk_secs,
        overlap_secs=args.overlap_secs,
        language=args.language,
        model=args.model,
        segmenter=args.segmenter,
        max_words=args.max_words,
        max_duration=args.max_duration,
        pause=args.pause,
        stdout=args.stdout,
    )
    try:
        return transcribe.transcribe_file(cfg)
    finally:
        if tmp_wav.exists():
            tmp_wav.unlink()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
