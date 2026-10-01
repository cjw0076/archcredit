import pytest
import torch

from archcredit import (
    ModelConfig,
    RNNArchitecture,
    apply_gradients,
    create_architecture,
    create_credit_rule,
    task_loss,
)
from archcredit.benchmarks import Batch


def batch(input_size=2, output_size=1):
    torch.manual_seed(4)
    return Batch(
        inputs=torch.randn(3, 5, input_size, dtype=torch.float64),
        targets=torch.randn(3, 5, output_size, dtype=torch.float64),
        mask=torch.tensor([0, 0, 1, 1, 1], dtype=torch.float64).view(1, 5, 1).expand(3, -1, -1),
    )


@pytest.mark.parametrize("architecture", ["rnn", "transformer"])
def test_bptt_matches_autograd(architecture):
    model = create_architecture(architecture, ModelConfig(2, 4, 1)).double()
    sample = batch()
    expected = torch.autograd.grad(
        task_loss(model(sample.inputs), sample), tuple(model.parameters())
    )
    actual = create_credit_rule("bptt").compute_gradients(model, sample)
    for (name, _), gradient in zip(model.named_parameters(), expected, strict=True):
        torch.testing.assert_close(actual[name], gradient)
        assert actual[name].grad_fn is None


def test_bptt_handles_frozen_and_unused_extension_parameters():
    class ExtendedRNN(RNNArchitecture):
        def __init__(self, config):
            super().__init__(config)
            self.unused = torch.nn.Parameter(torch.ones(2))
            self.readout.weight.requires_grad_(False)

    model = ExtendedRNN(ModelConfig(2, 4, 1)).double()
    sample = batch()
    gradients = create_credit_rule("bptt").compute_gradients(model, sample)
    assert gradients.keys() == dict(model.named_parameters()).keys()
    assert torch.count_nonzero(gradients["unused"]) == 0
    assert torch.count_nonzero(gradients["readout.weight"]) == 0
    assert torch.count_nonzero(gradients["input.weight"]) > 0
    apply_gradients(model, gradients, 0.1)
    torch.testing.assert_close(model.unused, torch.ones_like(model.unused))


def test_bptt_all_frozen_and_nondifferentiable_output():
    model = create_architecture("rnn", ModelConfig(2, 4, 1)).double()
    for value in model.parameters():
        value.requires_grad_(False)
    gradients = create_credit_rule("bptt").compute_gradients(model, batch())
    assert all(torch.count_nonzero(value) == 0 for value in gradients.values())

    class DetachedOutput(RNNArchitecture):
        def forward(self, inputs):
            return super().forward(inputs).detach()

    detached = DetachedOutput(ModelConfig(2, 4, 1)).double()
    gradients = create_credit_rule("bptt").compute_gradients(detached, batch())
    assert all(torch.count_nonzero(value) == 0 for value in gradients.values())


def test_scalar_eprop_matches_exact_gradient():
    model = create_architecture("rnn", ModelConfig(2, 1, 2)).double()
    sample = batch(output_size=2)
    exact = create_credit_rule("bptt").compute_gradients(model, sample)
    local = create_credit_rule("eprop").compute_gradients(model, sample)
    for name in exact:
        torch.testing.assert_close(local[name], exact[name], rtol=1e-10, atol=1e-12)


def test_single_step_multineuron_eprop_matches_exact_gradient():
    model = create_architecture("rnn", ModelConfig(2, 3, 1)).double()
    sample = Batch(
        torch.randn(2, 1, 2, dtype=torch.float64),
        torch.randn(2, 1, 1, dtype=torch.float64),
        torch.ones(2, 1, 1, dtype=torch.float64),
    )
    exact = create_credit_rule("bptt").compute_gradients(model, sample)
    local = create_credit_rule("eprop").compute_gradients(model, sample)
    for name in exact:
        torch.testing.assert_close(local[name], exact[name], rtol=1e-10, atol=1e-12)


def test_multineuron_eprop_independent_recurrence():
    model = create_architecture("rnn", ModelConfig(2, 3, 1)).double()
    sample = batch()
    actual = create_credit_rule("eprop").compute_gradients(model, sample)
    with torch.no_grad():
        prediction, hidden, previous = model.forward_with_trace(sample.inputs)
        dy = 2 * (prediction - sample.targets) * sample.mask / sample.mask.sum()
        for name, presynaptic in (("input.weight", sample.inputs), ("recurrent.weight", previous)):
            expected = torch.zeros_like(actual[name])
            for b in range(3):
                for j in range(3):
                    for i in range(presynaptic.shape[-1]):
                        eligibility = 0.0
                        for t in range(5):
                            eligibility = (1 - hidden[b, t, j] ** 2) * (
                                presynaptic[b, t, i] + model.recurrent.weight[j, j] * eligibility
                            )
                            expected[j, i] += dy[b, t, 0] * model.readout.weight[0, j] * eligibility
            torch.testing.assert_close(actual[name], expected, rtol=1e-10, atol=1e-12)
    # Approximation is explicit: non-diagonal recurrent paths differ from BPTT.
    exact = create_credit_rule("bptt").compute_gradients(model, sample)
    assert not torch.allclose(actual["input.weight"], exact["input.weight"])


@pytest.mark.parametrize("rule", ["eprop", "hebbian"])
def test_local_rule_update_and_unsupported_pair(rule):
    model = create_architecture("rnn", ModelConfig(2, 4, 1)).double()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    local = create_credit_rule(rule)
    gradients = local.compute_gradients(model, batch())
    assert gradients.keys() == before.keys()
    assert all(
        torch.isfinite(value).all() and value.grad_fn is None for value in gradients.values()
    )
    assert sum(value.abs().sum() for value in gradients.values()) > 0
    apply_gradients(model, gradients, lr=0.01)
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
    transformer = create_architecture("transformer", ModelConfig(2, 4, 1)).double()
    with pytest.raises(ValueError, match="supports only RNNArchitecture"):
        local.compute_gradients(transformer, batch())


def test_hebbian_hidden_is_local_readout_is_supervised():
    model = create_architecture("rnn", ModelConfig(2, 4, 1)).double()
    sample = batch()
    local = create_credit_rule("hebbian").compute_gradients(model, sample)
    exact = create_credit_rule("bptt").compute_gradients(model, sample)
    for name in ("readout.weight", "readout.bias"):
        torch.testing.assert_close(local[name], exact[name])
    changed = Batch(sample.inputs, sample.targets + 5, sample.mask)
    other = create_credit_rule("hebbian").compute_gradients(model, changed)
    for name in ("input.weight", "input.bias", "recurrent.weight"):
        torch.testing.assert_close(local[name], other[name])
    with torch.no_grad():
        _, hidden, _ = model.forward_with_trace(sample.inputs)
        expected = -(
            torch.einsum("bth,bti->hi", hidden, sample.inputs) / 15
            - hidden.square().mean(dim=(0, 1))[:, None] * model.input.weight
        )
        torch.testing.assert_close(local["input.weight"], expected)


@pytest.mark.parametrize("invalid", ["nan", "shape", "keys", "dtype"])
def test_invalid_update_is_atomic(invalid):
    model = create_architecture("rnn", ModelConfig(2, 4, 1)).double()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    gradients = {name: torch.ones_like(value) for name, value in before.items()}
    # Corrupt the last parameter to catch accidental partial application.
    name = list(gradients)[-1]
    if invalid == "nan":
        gradients[name].fill_(float("nan"))
    elif invalid == "shape":
        gradients[name] = torch.ones(5, dtype=torch.float64)
    elif invalid == "dtype":
        gradients[name] = gradients[name].float()
    else:
        del gradients[name]
    with pytest.raises(ValueError):
        apply_gradients(model, gradients, lr=0.1)
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_task_loss_mask_and_empty_mask():
    sample = batch()
    prediction = torch.zeros_like(sample.targets)
    expected = sample.targets[:, 2:].square().mean()
    torch.testing.assert_close(task_loss(prediction, sample), expected)
    empty = Batch(sample.inputs, sample.targets, torch.zeros_like(sample.mask))
    assert task_loss(prediction, empty).item() == 0
    two_dimensional = Batch(sample.inputs, sample.targets, sample.mask.squeeze(-1))
    torch.testing.assert_close(task_loss(prediction, two_dimensional), expected)


@pytest.mark.parametrize("shape", [(5,), (1, 5, 1), (3, 5, 2), (3, 5, 1, 1)])
def test_loss_and_local_rules_reject_ambiguous_masks(shape):
    model = create_architecture("rnn", ModelConfig(2, 4, 1)).double()
    sample = batch()
    invalid = Batch(sample.inputs, sample.targets, torch.ones(shape, dtype=torch.float64))
    with pytest.raises(ValueError, match="mask must have shape"):
        task_loss(model(sample.inputs), invalid)
    for name in ("eprop", "hebbian"):
        with pytest.raises(ValueError, match="mask must have shape"):
            create_credit_rule(name).compute_gradients(model, invalid)


def test_loss_rejects_target_broadcast():
    sample = batch()
    invalid = Batch(sample.inputs, sample.targets[:1], sample.mask)
    with pytest.raises(ValueError, match="same"):
        task_loss(torch.zeros_like(sample.targets), invalid)


@pytest.mark.parametrize("rule", ["bptt", "eprop"])
def test_empty_supervision_returns_zero_task_gradients(rule):
    model = create_architecture("rnn", ModelConfig(2, 4, 1)).double()
    sample = batch()
    empty = Batch(sample.inputs, sample.targets, torch.zeros_like(sample.mask))
    gradients = create_credit_rule(rule).compute_gradients(model, empty)
    assert all(torch.count_nonzero(value) == 0 for value in gradients.values())


@pytest.mark.parametrize("lr", [-0.1, 0.0, float("nan"), float("inf")])
def test_invalid_learning_rate_is_atomic(lr):
    model = create_architecture("rnn", ModelConfig(2, 4, 1))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    with pytest.raises(ValueError, match="lr"):
        apply_gradients(model, {name: torch.ones_like(value) for name, value in before.items()}, lr)
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_finite_update_overflow_is_atomic():
    model = create_architecture("rnn", ModelConfig(2, 4, 1))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    gradients = {name: torch.ones_like(value) for name, value in before.items()}
    with pytest.raises(ValueError, match="proposed update"):
        apply_gradients(model, gradients, lr=1e100)
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_apply_gradients_preserves_frozen_parameters():
    model = create_architecture("rnn", ModelConfig(2, 4, 1))
    model.input.weight.requires_grad_(False)
    before = model.input.weight.detach().clone()
    gradients = {name: torch.ones_like(value) for name, value in model.named_parameters()}
    apply_gradients(model, gradients, lr=0.1)
    torch.testing.assert_close(model.input.weight, before, rtol=0, atol=0)


def test_bptt_overfits_tiny_fixed_batch():
    torch.manual_seed(17)
    model = create_architecture("rnn", ModelConfig(1, 8, 1))
    inputs = torch.randn(4, 3, 1) * 0.5
    sample = Batch(inputs, inputs * 0.5, torch.ones(4, 3, 1))
    initial = task_loss(model(inputs), sample).item()
    rule = create_credit_rule("bptt")
    for _ in range(100):
        apply_gradients(model, rule.compute_gradients(model, sample), lr=0.1)
    assert task_loss(model(inputs), sample).item() < initial * 0.2
