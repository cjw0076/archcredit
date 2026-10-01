"""PyTorch-native open laboratory for architectures and credit assignment."""

from .architectures import RNNArchitecture, TransformerArchitecture
from .config import ModelConfig
from .credit import BPTT, EProp, Hebbian, apply_gradients
from .loss import task_loss
from .protocols import Architecture, CreditRule
from .registry import (
    create_architecture,
    create_credit_rule,
    list_architectures,
    list_credit_rules,
    register_architecture,
    register_credit_rule,
)
from .serialization import load_checkpoint, save_checkpoint

__version__ = "0.1.0"

__all__ = [
    "Architecture",
    "BPTT",
    "CreditRule",
    "EProp",
    "Hebbian",
    "ModelConfig",
    "RNNArchitecture",
    "TransformerArchitecture",
    "apply_gradients",
    "create_architecture",
    "create_credit_rule",
    "list_architectures",
    "list_credit_rules",
    "load_checkpoint",
    "register_architecture",
    "register_credit_rule",
    "save_checkpoint",
    "task_loss",
]
