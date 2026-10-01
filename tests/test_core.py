from dataclasses import asdict

import pytest
import torch
from torch import nn

from archcredit import (
    Architecture,
    CreditRule,
    ModelConfig,
    create_architecture,
    create_credit_rule,
    list_architectures,
    list_credit_rules,
    load_checkpoint,
    register_architecture,
    register_credit_rule,
    save_checkpoint,
)


@pytest.mark.parametrize(
    "field", ["input_size", "hidden_size", "output_size", "num_heads", "num_layers"]
)
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_config_rejects_invalid(field, invalid):
    values = asdict(ModelConfig(2, 8, 1))
    values[field] = invalid
    with pytest.raises(ValueError, match="positive integer"):
        ModelConfig(**values)


def test_config_roundtrip():
    config = ModelConfig(2, 8, 1)
    assert ModelConfig(**asdict(config)) == config


@pytest.mark.parametrize("name", ["rnn", "transformer"])
def test_reference_shape_and_protocol(name):
    model = create_architecture(name, ModelConfig(2, 8, 3))
    assert isinstance(model, Architecture)
    assert isinstance(create_credit_rule("bptt"), CreditRule)
    assert model(torch.randn(4, 7, 2)).shape == (4, 7, 3)


@pytest.mark.parametrize("name", ["rnn", "transformer"])
def test_causal_forward(name):
    model = create_architecture(name, ModelConfig(2, 8, 1)).eval()
    first = torch.randn(2, 6, 2)
    second = first.clone()
    second[:, 3:] += 10
    with torch.no_grad():
        torch.testing.assert_close(model(first)[:, :3], model(second)[:, :3])


@pytest.mark.parametrize("name", ["rnn", "transformer"])
def test_serialization_roundtrip(name, tmp_path):
    model = create_architecture(name, ModelConfig(2, 8, 1)).double().eval()
    inputs = torch.randn(2, 5, 2, dtype=torch.float64)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model)
    restored = load_checkpoint(path)
    assert restored.config == model.config
    assert restored.architecture_name == name
    assert not restored.training
    assert next(restored.parameters()).dtype == torch.float64
    torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)


def test_registry_extension_and_errors():
    @register_architecture("test_linear")
    class LinearArchitecture(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.layer = nn.Linear(config.input_size, config.output_size)

        def forward(self, inputs):
            return self.layer(inputs)

    @register_credit_rule("test_credit")
    class ZeroCredit:
        def compute_gradients(self, model, batch):
            return {name: torch.zeros_like(value) for name, value in model.named_parameters()}

    assert (
        create_architecture("test_linear", ModelConfig(2, 8, 1)).architecture_name == "test_linear"
    )
    assert isinstance(create_credit_rule("test_credit"), ZeroCredit)
    assert {"rnn", "transformer", "cosmos"} <= set(list_architectures())
    assert {"bptt", "eprop", "hebbian"} <= set(list_credit_rules())
    with pytest.raises(ValueError, match="already registered"):
        register_architecture("test_linear")(LinearArchitecture)
    with pytest.raises(ValueError, match="already registered"):
        register_credit_rule("test_credit")(ZeroCredit)
    with pytest.raises(ValueError, match="Unknown architecture"):
        create_architecture("missing", ModelConfig(2, 8, 1))
    with pytest.raises(ValueError, match="Unknown credit rule"):
        create_credit_rule("missing")
    with pytest.raises(NotImplementedError, match="experimental extension"):
        create_architecture("cosmos", ModelConfig(2, 8, 1))


def test_transformer_head_validation():
    with pytest.raises(ValueError, match="divisible"):
        create_architecture("transformer", ModelConfig(2, 7, 1))


def test_checkpoint_rejects_metadata_and_weight_mismatch(tmp_path):
    path = tmp_path / "invalid.pt"
    torch.save({"format_version": 2}, path)
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(path)
    torch.save({"format_version": 1, "architecture": 123, "config": {}}, path)
    with pytest.raises(ValueError, match="architecture name"):
        load_checkpoint(path)
    model = create_architecture("rnn", ModelConfig(2, 8, 1))
    save_checkpoint(path, model)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["config"]["hidden_size"] = 4
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_checkpoint(path)


def test_checkpoint_parameterless_extension(tmp_path):
    @register_architecture("test_parameterless")
    class IdentityArchitecture(nn.Module):
        def __init__(self, config):
            super().__init__()

        def forward(self, inputs):
            return inputs

    model = create_architecture("test_parameterless", ModelConfig(1, 1, 1))
    path = tmp_path / "parameterless.pt"
    save_checkpoint(path, model)
    restored = load_checkpoint(path)
    inputs = torch.randn(1, 3, 1)
    torch.testing.assert_close(restored(inputs), inputs)


def test_checkpoint_preserves_mixed_state_dtypes(tmp_path):
    model = create_architecture("rnn", ModelConfig(2, 4, 1))
    model.input.double()
    model.recurrent.half()
    path = tmp_path / "mixed.pt"
    save_checkpoint(path, model)
    restored = load_checkpoint(path)
    for name, value in model.state_dict().items():
        other = restored.state_dict()[name]
        assert value.dtype == other.dtype
        torch.testing.assert_close(value, other, rtol=0, atol=0)
