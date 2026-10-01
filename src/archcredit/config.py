"""Serializable reference architecture configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    input_size: int
    hidden_size: int
    output_size: int
    num_heads: int = 2
    num_layers: int = 1

    def __post_init__(self) -> None:
        for name in ("input_size", "hidden_size", "output_size", "num_heads", "num_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
