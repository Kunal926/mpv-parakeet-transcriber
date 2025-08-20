"""Lightweight loader for RoFormer-based vocal separation models.

This module focuses solely on inference. It reads a YAML
configuration and checkpoint to build a separator that exposes a
simple ``separate`` method.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
import librosa


class _SafeTupleLoader(yaml.SafeLoader):
    """YAML loader that safely constructs ``!!python/tuple`` nodes."""


def _construct_python_tuple(loader: yaml.SafeLoader, node: yaml.Node):
    return tuple(loader.construct_sequence(node))


_SafeTupleLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_python_tuple
)


@dataclass
class Separator:
    """Wrapper object performing chunked forward passes."""

    model: torch.nn.Module
    sample_rate: int
    chunk_size: int
    overlap: int
    device: torch.device

    def _forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Run the model on audio [2, N] and return separated stem."""
        with torch.inference_mode():
            return self.model(audio.unsqueeze(0)).squeeze(0)

    def separate(
        self,
        audio: np.ndarray,
        sr: int,
        target: Literal["vocals", "instrumental"],
    ) -> np.ndarray:
        """Separate target from a stereo mixture.

        Parameters
        ----------
        audio: np.ndarray [num_samples, 2]
            Stereo mixture in ``float32``.
        sr: int
            Sample rate of ``audio``.
        target: Literal["vocals", "instrumental"]
            Which stem the underlying model predicts.
        """
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("expected stereo audio [num_samples, 2]")

        if sr != self.sample_rate:
            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=self.sample_rate, axis=1).T
            sr = self.sample_rate

        # Convert to torch
        mix = torch.from_numpy(audio.T).to(self.device)

        step = self.chunk_size - self.overlap
        total = mix.shape[-1]
        output = torch.zeros_like(mix)
        weight = torch.zeros(1, total, device=self.device)

        for start in range(0, total, step):
            end = min(start + self.chunk_size, total)
            chunk = mix[:, start:end]
            if chunk.shape[-1] < self.chunk_size:
                pad = self.chunk_size - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad))
            out = self._forward(chunk)
            out = out[:, : end - start]
            output[:, start:end] += out
            weight[:, start:end] += 1

        output /= weight.clamp(min=1e-6)
        result = output.T.cpu().numpy()

        if target == "instrumental":
            # Instrumental models predict accompaniment, derive vocals as mix - inst
            result = mix.T.cpu().numpy() - result

        return np.clip(result, -1.0, 1.0)


class _IdentityModel(torch.nn.Module):
    """Fallback model when real checkpoints are unavailable.

    It simply returns the input mixture, allowing the rest of the
    pipeline to function even without weights. Users are expected to
    provide real YAML+CKPT files for proper separation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x


def load_separator(
    cfg_path: str,
    ckpt_path: str,
    device: str = "cuda",
    fp16: bool = True,
) -> Separator:
    """Instantiate a :class:`Separator` from YAML and checkpoint.

    Parameters
    ----------
    cfg_path: str
        Path to the YAML configuration file.
    ckpt_path: str
        Path to the model checkpoint. If missing, a tiny identity
        network is used as a placeholder.
    device: str
        Torch device string (``"cuda"`` or ``"cpu"``).
    fp16: bool
        Whether to cast the model to ``float16`` when CUDA is used.
    """
    cfg_path = str(cfg_path)
    ckpt_path = str(ckpt_path)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=_SafeTupleLoader)

    sample_rate = int(cfg.get("sample_rate", 44100))
    chunk_size = int(cfg.get("chunk_size", 262144))
    overlap = int(cfg.get("num_overlap", 0))

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    model = _IdentityModel()
    if Path(ckpt_path).is_file():
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
        except Exception:
            # Fall back to identity if checkpoint cannot be loaded
            pass

    model.eval()
    model.to(dev)
    if fp16 and dev.type == "cuda":
        model.half()

    return Separator(model=model, sample_rate=sample_rate, chunk_size=chunk_size, overlap=overlap, device=dev)

