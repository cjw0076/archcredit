"""Credit geometry diagnostics, with explicit limits on causal interpretation."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .benchmarks import Batch
from .protocols import CreditRule


@dataclass
class CreditAlignment:
    """Pooled rho = E[u dot (-g)] / sqrt(E[||u||²] E[||g||²]).

    Inputs are update directions, not positive loss gradients. Zero directions
    make rho undefined. Pooling sufficient statistics avoids mean-cosine bias.
    """

    dot: float = 0.0
    update_squared: float = 0.0
    reference_squared: float = 0.0
    samples: int = 0

    def update(self, update: Mapping[str, Tensor], reference: Mapping[str, Tensor]) -> None:
        if set(update) != set(reference):
            raise ValueError("Update and reference must have identical parameter names")
        for name in sorted(update):
            u = update[name].detach().double()
            r = reference[name].detach().double()
            if u.shape != r.shape:
                raise ValueError(f"Parameter shape differs for {name}")
            if not torch.isfinite(u).all() or not torch.isfinite(r).all():
                raise ValueError("Credit directions must be finite")
            self.dot += float((u * r).sum())
            self.update_squared += float(u.square().sum())
            self.reference_squared += float(r.square().sum())
        self.samples += 1

    @property
    def rho(self) -> float | None:
        denominator = math.sqrt(self.update_squared * self.reference_squared)
        return max(-1.0, min(1.0, self.dot / denominator)) if denominator else None


def measure_alignment(model: nn.Module, rule: CreditRule, batch: Batch) -> float | None:
    """Compare same-batch credit with BPTT without applying parameter updates.

    The caller supplies a deterministic architecture (the references have no
    dropout). CPU random state is restored around custom credit computations.
    """
    from .credit import BPTT

    metric = CreditAlignment()
    with torch.random.fork_rng(devices=[]):
        initial_state = torch.random.get_rng_state()
        candidate = rule.compute_gradients(model, batch)
        torch.random.set_rng_state(initial_state)
        reference = BPTT().compute_gradients(model, batch)
        metric.update(
            {name: -gradient for name, gradient in candidate.items()},
            {name: -gradient for name, gradient in reference.items()},
        )
    return metric.rho


def sweep_horizons(
    model: nn.Module,
    rule: CreditRule,
    horizons: Sequence[int],
    batch_factory: Callable[[int], Batch],
    threshold: float = 0.5,
) -> dict[str, object]:
    """Measure a fixed model at supplied delays, returning an alignment proxy.

    ``batch_factory`` should use its own generator; no training occurs here.
    """
    scores = [measure_alignment(model, rule, batch_factory(horizon)) for horizon in horizons]
    return summarize_horizon(horizons, scores, threshold)


def summarize_horizon(
    horizons: Sequence[int], scores: Sequence[float | None], threshold: float = 0.5
) -> dict[str, object]:
    """Scaffold: largest contiguous passing *measured* alignment horizon.

    A missing or failing score stops the prefix. Unmeasured delays are never
    inferred; this alignment-based proxy is not causal long-range task proof.
    """
    if len(horizons) != len(scores) or not horizons:
        raise ValueError("horizons and scores must be nonempty and have equal lengths")
    if list(horizons) != sorted(set(horizons)) or any(h < 1 for h in horizons):
        raise ValueError("horizons must be positive, unique, and increasing")
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("threshold must be a finite alignment in [-1, 1]")
    effective = None
    for horizon, score in zip(horizons, scores, strict=True):
        if score is None or not math.isfinite(score) or score < threshold:
            break
        effective = horizon
    return {
        "kind": "alignment-based proxy; not causal task-performance proof",
        "effective_measured_horizon": effective,
        "threshold": threshold,
        "measured_horizons": list(horizons),
        "scores": list(scores),
    }
