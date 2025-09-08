"""Vocal separation orchestration."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

__all__ = ["separate", "main"]


def separate(
    input_wav: Path,
    output_wav: Path,
    cfg: Path,
    ckpt: Path,
    target: str = "vocals",
    device: str = "cuda",
    fp16: bool = False,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "separation.bsr_separate",
        "--in_wav",
        str(input_wav),
        "--out_wav",
        str(output_wav),
        "--cfg",
        str(cfg),
        "--ckpt",
        str(ckpt),
        "--target",
        target,
        "--device",
        device,
    ]
    if fp16:
        cmd.append("--fp16")
    subprocess.run(cmd, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="parakeet-separate")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--cfg", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--target", default="vocals")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--fp16", action="store_true")
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    separate(args.input, args.output, args.cfg, args.ckpt, args.target, args.device, args.fp16)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
