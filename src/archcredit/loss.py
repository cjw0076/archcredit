"""One masked objective shared by exact and local credit rules."""

from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from .benchmarks import Batch


def _time_mask(prediction: Tensor, batch: "Batch") -> Tensor:
    if prediction.ndim != 3 or prediction.shape != batch.targets.shape:
        raise ValueError("prediction and targets must have the same [B, T, O] shape")
    mask = batch.mask.unsqueeze(-1) if batch.mask.ndim == 2 else batch.mask
    if mask.shape != (*prediction.shape[:2], 1):
        raise ValueError("mask must have shape [B, T] or [B, T, 1]")
    return mask


def task_loss(prediction: Tensor, batch: "Batch") -> Tensor:
    """Mean squared error across supervised steps and outputs; empty mask gives zero."""
    mask = _time_mask(prediction, batch)
    return ((prediction - batch.targets).square() * mask).sum() / (
        mask.sum().clamp_min(1) * prediction.shape[-1]
    )
