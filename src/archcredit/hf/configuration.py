"""Numeric sequence configuration, following Transformers custom-model conventions."""

from dataclasses import asdict
from typing import Any

from transformers import PretrainedConfig

from archcredit.config import ModelConfig


class ArchCreditConfig(PretrainedConfig):
    model_type = "archcredit"

    def __init__(
        self,
        architecture: str = "rnn",
        model_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if architecture not in {"rnn", "transformer"}:
            raise ValueError("HF scaffold supports rnn and transformer reference architectures")
        self.architecture = architecture
        default = ModelConfig(input_size=2, hidden_size=8, output_size=1)
        self.model_config = asdict(ModelConfig(**model_config)) if model_config else asdict(default)
