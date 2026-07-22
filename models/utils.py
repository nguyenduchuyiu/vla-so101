"""Shared inference helpers for SmolVLM-VLA evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

from .modeling_smolvlm_vla import SmolVLMVLA
from .processing_smolvlm_vla import SmolVLMVLAProcessor


def pick_device() -> torch.device:
    """cuda if available, else mps, else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_vla_for_inference(
    checkpoint: Path, device: torch.device
) -> tuple[SmolVLMVLA, SmolVLMVLAProcessor]:
    """Load a SmolVLM-VLA checkpoint for inference.

    MPS is forced to FP32 because Metal is unstable with this checkpoint's
    mixed FP16 matmuls.
    """
    config = SmolVLMVLA.config_class.from_pretrained(checkpoint)
    if device.type == "mps":
        config.vlm_dtype = "float32"
    model = SmolVLMVLA.from_pretrained(
        checkpoint,
        config=config,
        dtype=torch.float32 if device.type == "mps" else None,
    ).to(device).eval()
    processor = SmolVLMVLAProcessor.from_pretrained(model.config.smolvlm_model_path)
    return model, processor