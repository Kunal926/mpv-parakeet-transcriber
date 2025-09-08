"""NeMo ASR wrapper for Parakeet."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import sys

import torch

from .srt_utils import format_time_srt

try:  # Lazy import
    import nemo.collections.asr as nemo_asr
except Exception:  # pragma: no cover
    nemo_asr = None  # type: ignore

__all__ = ["TranscribeConfig", "transcribe_file"]


@dataclass
class TranscribeConfig:
    audio_path: Path
    output_path: Optional[Path]
    audio_start_offset: float = 0.0
    precision: str = "auto"
    device: str = "auto"
    batch_size: int = 16
    chunk_secs: float = 30.0
    overlap_secs: float = 5.0
    language: str = "en"
    model: str = "nvidia/parakeet-tdt-0.6b-v2"
    segmenter: str = "word"
    max_words: int = 12
    max_duration: float = 6.0
    pause: float = 0.6
    stdout: bool = False


def _dtype_from_precision(precision: str) -> Optional[torch.dtype]:
    return {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(precision)


def _device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def transcribe_file(cfg: TranscribeConfig) -> int:
    """Transcribe ``cfg.audio_path`` into ``cfg.output_path`` or stdout."""
    if nemo_asr is None:
        raise RuntimeError("nemo_toolkit is not available")

    device = _device(cfg.device)
    dtype = _dtype_from_precision(cfg.precision)
    torch.cuda.empty_cache()

    model = nemo_asr.models.ASRModel.from_pretrained(cfg.model, map_location=device)
    model.eval()

    audio_list = [str(cfg.audio_path)]
    try:
        with torch.autocast(device.type, enabled=dtype is not None, dtype=dtype):
            hyps = model.transcribe(
                paths2audio_files=audio_list,
                batch_size=cfg.batch_size,
                num_workers=0,
                language=cfg.language,
                return_hypotheses=True,
                timestamp_type=cfg.segmenter,
            )[1]
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover
        msg = "CUDA out of memory"
        if cfg.output_path:
            cfg.output_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n[CUDA OOM]\n\n", encoding="utf-8"
            )
        raise RuntimeError(msg) from exc

    processed: List[dict] = []
    for hyp in hyps:
        if cfg.segmenter == "word" and getattr(hyp, "timestep", None) is not None:
            for word, ts in zip(hyp.words, hyp.timestep):
                processed.append(
                    {
                        "word_from_model": word,
                        "start_seconds": ts.start + cfg.audio_start_offset,
                        "end_seconds": ts.end + cfg.audio_start_offset,
                    }
                )
        else:
            for ts in hyp.timestep:
                processed.append(
                    {
                        "text_from_model": ts.text,
                        "start_seconds": ts.start + cfg.audio_start_offset,
                        "end_seconds": ts.end + cfg.audio_start_offset,
                    }
                )

    out = sys.stdout if cfg.stdout else cfg.output_path.open("w", encoding="utf-8") if cfg.output_path else None
    if out is None:
        return 0
    try:
        _write_srt(processed, out, cfg.segmenter, cfg)
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


def _write_srt(data: List[dict], out, segmenter: str, cfg: TranscribeConfig) -> None:
    if segmenter == "word":
        seg_num = 1
        current: List[str] = []
        start = 0.0
        last_end = 0.0

        def flush(final_end: float) -> None:
            nonlocal seg_num, current, start
            if not current:
                return
            out.write(
                f"{seg_num}\n{format_time_srt(start)} --> {format_time_srt(final_end)}\n" + " ".join(current) + "\n\n"
            )
            seg_num += 1
            current = []
            start = 0.0

        for item in data:
            word = item.get("word_from_model", "")
            s = item.get("start_seconds", 0.0)
            e = item.get("end_seconds", 0.0)
            if not current:
                current.append(word)
                start = s
            else:
                gap = s - last_end
                duration = e - start
                if gap >= cfg.pause or len(current) >= cfg.max_words or duration > cfg.max_duration:
                    flush(last_end)
                    current = [word]
                    start = s
                else:
                    current.append(word)
            last_end = e
        if current:
            flush(last_end)
    else:
        for idx, item in enumerate(data, 1):
            text = item.get("text_from_model", "")
            s = item.get("start_seconds", 0.0)
            e = item.get("end_seconds", s)
            out.write(f"{idx}\n{format_time_srt(s)} --> {format_time_srt(e)}\n{text}\n\n")
    out.flush()
