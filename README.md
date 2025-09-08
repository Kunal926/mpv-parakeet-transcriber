# MPV Parakeet Transcriber

Utilities for generating subtitles in [MPV](https://mpv.io) using NVIDIA's Parakeet ASR.
The repository ships a Python package and Lua helpers which can transcribe the
currently playing file, optionally isolating vocals with RoFormer models.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .[separation]
parakeet-transcribe path/to/audio.wav --stdout
```

Place RoFormer presets and checkpoints under `weights/roformer/`.  Only small
configuration files are tracked; obtain the actual model weights from their
respective projects.

## Prerequisites

* MPV media player
* FFmpeg and FFprobe in `PATH`
* Python 3.10+
* (Recommended) CUDA capable GPU for real‑time use

## Installation

1. Clone this repo and install the package:
   ```bash
   git clone https://github.com/Kunal926/mpv-parakeet-transcriber.git
   cd mpv-parakeet-transcriber
   pip install -e .[separation]
   ```
2. Copy `parakeet_caption.lua` to MPV's `scripts/` directory.  Adjust the
   configuration table at the top for your environment.

## CLI Usage

```
parakeet-transcribe <media> [--output subs.srt] [--stdout]
```

Key options:

* `--precision {auto,float32,bfloat16,float16}` – arithmetic precision
* `--ffmpeg-filters "loudnorm=I=-16"` – preprocessing filter chain
* `--device {auto,cuda,cpu}` – execution device
* `--language` and `--model` – NeMo model parameters

`parakeet-separate` exposes the RoFormer vocal isolation pipeline and accepts
`--cfg` and `--ckpt` arguments pointing to the YAML and checkpoint files.

## MPV keybindings

* `Alt+4` – default transcription
* `Alt+5` – force float32
* `Alt+6` – FFmpeg preprocessing
* `Alt+7` – preprocessing + float32
* `Alt+8` – fast vocal isolation then ASR
* `Alt+9` – high‑quality isolation then ASR

## Troubleshooting

* CUDA out of memory → lower `--batch-size`, switch precision, or run on CPU.
* Missing weights → download RoFormer checkpoints and place them under
  `weights/roformer/` as documented above.

## License

This project and accompanying Lua scripts are licensed under
[CC‑BY‑4.0](LICENSE).
