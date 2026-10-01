"""CPU graph capture and autograd parity; no performance claim."""

import copy

import pytest
import torch

from archcredit import ModelConfig, create_architecture


@pytest.mark.parametrize("name", ["rnn", "transformer"])
def test_compile_forward_backward_parity(name):
    torch.manual_seed(3)
    eager = create_architecture(name, ModelConfig(input_size=2, hidden_size=8, output_size=1))
    captured = copy.deepcopy(eager)
    compiled = torch.compile(captured, backend="aot_eager", fullgraph=True)
    inputs = torch.randn(2, 5, 2)
    expected = eager(inputs)
    actual = compiled(inputs)
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for reference, traced in zip(eager.parameters(), captured.parameters()):
        assert reference.grad is not None and traced.grad is not None
        torch.testing.assert_close(traced.grad, reference.grad)
