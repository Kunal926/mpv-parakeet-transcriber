from __future__ import annotations

from pathlib import Path
from typing import List

from mpv_parakeet import cli


def _noop(*_args, **_kwargs):
    return 0

def fake_transcribe(cfg):
    if cfg.output_path:
        Path(cfg.output_path).write_text(
            "1\n00:00:00,000 --> 00:00:00,500\nhello\n\n", encoding="utf-8"
        )
    return 0

def test_cli_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.transcribe, "transcribe_file", fake_transcribe)
    monkeypatch.setattr(cli.ffmpeg_io, "extract_audio", _noop)
    monkeypatch.setattr(cli.ffmpeg_io, "probe_start", lambda *_args, **_kwargs: 0.0)
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"")
    out = tmp_path / "out.srt"
    argv: List[str] = [str(sample), "--output", str(out)]
    assert cli.main(argv) == 0
    assert out.exists()
