"""Decorator registration for an architecture or credit rule in a few lines."""

from collections.abc import Callable
from typing import TypeVar

from torch import nn

from .config import ModelConfig
from .protocols import CreditRule

ArchitectureFactory = Callable[[ModelConfig], nn.Module]
CreditFactory = Callable[[], CreditRule]
_architectures: dict[str, ArchitectureFactory] = {}
_credits: dict[str, CreditFactory] = {}
AF = TypeVar("AF", bound=ArchitectureFactory)
CF = TypeVar("CF", bound=CreditFactory)


def register_architecture(name: str) -> Callable[[AF], AF]:
    def decorator(factory: AF) -> AF:
        if not name or name in _architectures:
            raise ValueError(f"Architecture name empty or already registered: {name!r}")
        _architectures[name] = factory
        return factory

    return decorator


def register_credit_rule(name: str) -> Callable[[CF], CF]:
    def decorator(factory: CF) -> CF:
        if not name or name in _credits:
            raise ValueError(f"Credit name empty or already registered: {name!r}")
        _credits[name] = factory
        return factory

    return decorator


def create_architecture(name: str, config: ModelConfig) -> nn.Module:
    try:
        factory = _architectures[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown architecture {name!r}; available: {list_architectures()}"
        ) from error
    model = factory(config)
    model.config = config
    model.architecture_name = name
    return model


def create_credit_rule(name: str) -> CreditRule:
    try:
        factory = _credits[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown credit rule {name!r}; available: {list_credit_rules()}"
        ) from error
    return factory()


def list_architectures() -> list[str]:
    return sorted(_architectures)


def list_credit_rules() -> list[str]:
    return sorted(_credits)
