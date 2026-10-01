"""Optional Hugging Face adapter; install archcredit[hf] to import this module."""

from .configuration import ArchCreditConfig
from .modeling import ArchCreditModel, register_auto_classes

__all__ = ["ArchCreditConfig", "ArchCreditModel", "register_auto_classes"]
