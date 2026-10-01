"""Small structural contracts; implementations remain native PyTorch modules."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from torch import Tensor, nn

from .config import ModelConfig

if TYPE_CHECKING:
    from .benchmarks import Batch


@runtime_checkable
class Architecture(Protocol):
    config: ModelConfig
    architecture_name: str

    def forward(self, inputs: Tensor) -> Tensor: ...

    def __call__(self, inputs: Tensor) -> Tensor: ...


@runtime_checkable
class CreditRule(Protocol):
    def compute_gradients(self, model: nn.Module, batch: "Batch") -> dict[str, Tensor]: ...
