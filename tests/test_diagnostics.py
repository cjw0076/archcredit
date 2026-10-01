import math

import pytest
import torch

from archcredit.benchmarks import generate_batch
from archcredit.config import ModelConfig
from archcredit.diagnostics import (
    CreditAlignment,
    measure_alignment,
    summarize_horizon,
    sweep_horizons,
)
from archcredit.registry import create_architecture, create_credit_rule


@pytest.mark.parametrize("direction,expected", [(1, 1), (-1, -1)])
def test_alignment_sign(direction, expected):
    metric = CreditAlignment()
    metric.update(
        {"x": torch.tensor([float(direction), 2.0 * direction])}, {"x": torch.tensor([1.0, 2.0])}
    )
    assert metric.rho == pytest.approx(expected)


def test_alignment_zero_is_undefined():
    metric = CreditAlignment()
    assert metric.rho is None
    metric.update({"x": torch.zeros(2)}, {"x": torch.ones(2)})
    assert metric.rho is None


def test_alignment_pools_sufficient_statistics_not_cosines():
    metric = CreditAlignment()
    metric.update({"x": torch.tensor([1.0])}, {"x": torch.tensor([1.0])})
    metric.update({"x": torch.tensor([-10.0])}, {"x": torch.tensor([1.0])})
    assert metric.rho == pytest.approx(-9 / math.sqrt(101 * 2))
    assert metric.samples == 2


def test_alignment_rejects_parameter_mismatch():
    with pytest.raises(ValueError):
        CreditAlignment().update({"x": torch.ones(1)}, {"y": torch.ones(1)})


def test_horizon_is_conservative_measured_prefix():
    result = summarize_horizon([2, 4, 8, 16], [0.9, 0.6, 0.2, 0.9])
    assert result["effective_measured_horizon"] == 4
    assert "not causal" in result["kind"]
    assert summarize_horizon([2, 4], [None, 0.9])["effective_measured_horizon"] is None


def test_horizon_requires_ordered_measurements():
    with pytest.raises(ValueError):
        summarize_horizon([4, 2], [0.8, 0.8])


def test_alignment_and_sweep_leave_model_and_rng_untouched():
    model = create_architecture("rnn", ModelConfig(1, 4, 1))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    state = torch.random.get_rng_state().clone()
    rule = create_credit_rule("bptt")

    def batches(horizon):
        return generate_batch("delayed-xor", 4, horizon, torch.Generator().manual_seed(4))

    assert measure_alignment(model, rule, batches(2)) == pytest.approx(1)
    result = sweep_horizons(model, rule, [2, 4], batches)
    assert result["effective_measured_horizon"] == 4
    assert torch.equal(state, torch.random.get_rng_state())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(before[name], value)
    assert all(parameter.grad is None for parameter in model.parameters())
