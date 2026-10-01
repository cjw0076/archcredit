"""Safe state-dict checkpoints; extension factories must be registered to load."""

from dataclasses import asdict
from os import PathLike

import torch
from torch import nn

from .config import ModelConfig
from .registry import create_architecture


def save_checkpoint(path: str | PathLike[str], model: nn.Module) -> None:
    config = getattr(model, "config", None)
    name = getattr(model, "architecture_name", None)
    if not isinstance(config, ModelConfig) or not isinstance(name, str):
        raise ValueError("Model needs ModelConfig config and a registered architecture_name")
    torch.save(
        {
            "format_version": 1,
            "architecture": name,
            "config": asdict(config),
            "training": model.training,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str | PathLike[str]) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if not isinstance(checkpoint.get("architecture"), str) or not isinstance(
        checkpoint.get("config"), dict
    ):
        raise ValueError("Checkpoint requires an architecture name and config dictionary")
    model = create_architecture(checkpoint["architecture"], ModelConfig(**checkpoint["config"]))
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.train(checkpoint.get("training", True))
    return model
