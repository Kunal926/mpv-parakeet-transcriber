from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any, Dict

import numpy as np
import torch
import yaml
import librosa
from torch.cuda.amp import autocast

# --- tiny YAML helper for !!python/tuple ---
class _SafeTupleLoader(yaml.SafeLoader): ...
def _construct_python_tuple(loader: yaml.SafeLoader, node: yaml.Node):
    return tuple(loader.construct_sequence(node))
_SafeTupleLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)

# ---- import the vendored ZFTurbo model code ----
# We try both common class names to be resilient to upstream naming.
def _import_melband_class():
    try:
        from separation.roformer_models.mel_band_roformer import MelBandRoformer
        return MelBandRoformer
    except Exception:
        try:
            from separation.roformer_models.mel_band_roformer import MelBand_Roformer as MelBandRoformer
            return MelBandRoformer
        except Exception as e:
            raise ImportError(
                "Could not import MelBand RoFormer class. "
                "Ensure separation/roformer_models/mel_band_roformer.py is present."
            ) from e

def _import_bs_class():
    try:
        from separation.roformer_models.bs_roformer import BSRoformer
        return BSRoformer
    except Exception:
        try:
            from separation.roformer_models.bs_roformer import BS_Roformer as BSRoformer
            return BSRoformer
        except Exception as e:
            raise ImportError(
                "Could not import BS-RoFormer class. "
                "Ensure separation/roformer_models/bs_roformer.py is present."
            ) from e

@dataclass
class Separator:
    model: torch.nn.Module
    sample_rate: int
    chunk_size: int
    overlap: int
    device: torch.device
    fp16: bool

    @torch.inference_mode()
    def _forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio shape: [2, T]
        # ZFTurbo models accept stereo batch: [B, 2, T]
        with autocast(enabled=self.fp16 and self.device.type == "cuda"):
            out = self.model(audio.unsqueeze(0))
        return out.squeeze(0)

    def separate(self, audio: np.ndarray, sr: int, target: Literal["vocals", "instrumental"]) -> np.ndarray:
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("expected stereo audio [num_samples, 2]")
        if sr != self.sample_rate:
            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=self.sample_rate, axis=1).T
            sr = self.sample_rate

        mix = torch.from_numpy(audio.T).to(self.device)  # [2, T]
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
            pred = self._forward(chunk)[:, : end - start]
            output[:, start:end] += pred
            weight[:, start:end] += 1

        output /= weight.clamp(min=1e-6)
        result = output.T.cpu().numpy()

        if target == "instrumental":
            result = mix.T.cpu().numpy() - result
        return np.clip(result, -1.0, 1.0)

def _build_melband(cfg_model: Dict[str, Any]) -> torch.nn.Module:
    MelBand = _import_melband_class()
    # Map common mel-band config keys from YAML (like your karaoke YAML)
    # Unknown args are omitted to stay compatible across forks.
    kwargs = {
        "dim":               int(cfg_model.get("dim", 384)),
        "depth":             int(cfg_model.get("depth", 6)),
        "num_bands":         int(cfg_model.get("num_bands", 60)),
        "dim_head":          int(cfg_model.get("dim_head", 64)),
        "heads":             int(cfg_model.get("heads", 8)),
        "attn_dropout":      float(cfg_model.get("attn_dropout", 0.0)),
        "ff_dropout":        float(cfg_model.get("ff_dropout", 0.0)),
        "num_stems":         int(cfg_model.get("num_stems", 1)),
        "stereo":            bool(cfg_model.get("stereo", True)),
        "time_transformer_depth": int(cfg_model.get("time_transformer_depth", 1)),
        "freq_transformer_depth": int(cfg_model.get("freq_transformer_depth", 1)),
        "mask_estimator_depth":   int(cfg_model.get("mask_estimator_depth", 2)),
        "flash_attn":        bool(cfg_model.get("flash_attn", True)),
        # STFT / Mel params expected by many forks
        "dim_freqs_in":      int(cfg_model.get("dim_freqs_in", 1025)),
        "sample_rate":       int(cfg_model.get("sample_rate", 44100)),
        "stft_n_fft":        int(cfg_model.get("stft_n_fft", 2048)),
        "stft_hop_length":   int(cfg_model.get("stft_hop_length", 441)),
        "stft_win_length":   int(cfg_model.get("stft_win_length", 2048)),
        "stft_normalized":   bool(cfg_model.get("stft_normalized", False)),
    }
    # only pass args actually in the constructor
    import inspect
    sig = inspect.signature(MelBand.__init__)
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return MelBand(**kwargs)

def _build_bs(cfg_model: Dict[str, Any]) -> torch.nn.Module:
    BS = _import_bs_class()
    import inspect
    # keep minimal set that is common across BS‑RoFormer forks
    kwargs = {k: cfg_model[k] for k in ("dim", "depth") if k in cfg_model}
    kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(BS.__init__).parameters}
    return BS(**kwargs)

def load_separator(cfg_path: str, ckpt_path: str, device: str = "cuda", fp16: bool = True) -> Separator:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=_SafeTupleLoader)

    audio_cfg   = cfg.get("audio", {})
    model_cfg   = cfg.get("model", {})
    # Heuristic: mel‑band configs have 'num_bands' and STFT params present (as in your karaoke YAML)
    is_melband  = "num_bands" in model_cfg or "dim_freqs_in" in model_cfg

    sample_rate = int(audio_cfg.get("sample_rate", model_cfg.get("sample_rate", 44100)))
    chunk_size  = int(audio_cfg.get("chunk_size", 262144))
    overlap     = int(cfg.get("inference", {}).get("num_overlap", audio_cfg.get("num_overlap", 0)))

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    if is_melband:
        model = _build_melband(model_cfg)
    else:
        model = _build_bs(model_cfg)

    model.eval().to(dev)
    if fp16 and dev.type == "cuda":
        try:
            model.half()
        except Exception:
            pass

    # Load checkpoint (support both raw state_dict and lightning checkpoints)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[RoFormer] Warning: unexpected keys in state_dict: {len(unexpected)} (model variant mismatch?)", flush=True)
    if missing:
        print(f"[RoFormer] Warning: missing keys in state_dict: {len(missing)} (non-critical layers may be re‑initialized).", flush=True)

    return Separator(model=model, sample_rate=sample_rate, chunk_size=chunk_size, overlap=overlap, device=dev, fp16=fp16)
