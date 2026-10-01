"""Local AutoModel adapter. Hub consumers must install archcredit[hf]."""

from typing import Any

from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from archcredit.config import ModelConfig
from archcredit.registry import create_architecture

from .configuration import ArchCreditConfig


class ArchCreditModel(PreTrainedModel):
    config_class = ArchCreditConfig
    base_model_prefix = "architecture"

    def __init__(self, config: ArchCreditConfig) -> None:
        super().__init__(config)
        self.architecture = create_architecture(
            config.architecture, ModelConfig(**config.model_config)
        )
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Keep the reference architecture's native PyTorch initialization."""
        return None

    def forward(self, inputs: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        """Accept numeric [batch, time, input_size] inputs; this is not a language model."""
        return {"logits": self.architecture(inputs)}


def register_auto_classes() -> None:
    """Explicit, repeatable local registration; no global registration on core import."""
    AutoConfig.register(ArchCreditConfig.model_type, ArchCreditConfig, exist_ok=True)
    AutoModel.register(ArchCreditConfig, ArchCreditModel, exist_ok=True)
