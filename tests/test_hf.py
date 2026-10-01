"""Optional safe-serialization and local AutoModel round trip."""

from dataclasses import asdict

import pytest
import torch

pytest.importorskip("transformers")

from transformers import AutoConfig, AutoModel  # noqa: E402

from archcredit.config import ModelConfig  # noqa: E402
from archcredit.hf import (  # noqa: E402
    ArchCreditConfig,
    ArchCreditModel,
    register_auto_classes,
)


@pytest.mark.parametrize("architecture", ["rnn", "transformer"])
def test_safe_hf_roundtrip(tmp_path, architecture):
    config = ArchCreditConfig(
        architecture=architecture,
        model_config=asdict(ModelConfig(input_size=2, hidden_size=8, output_size=1)),
    )
    torch.manual_seed(42)
    expected_model = ArchCreditModel(config)
    torch.manual_seed(42)
    model = ArchCreditModel(config).eval()
    for name, parameter in expected_model.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], parameter)
    inputs = torch.randn(2, 5, 2)
    expected = model(inputs)["logits"].detach()
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / "model.safetensors").is_file()
    register_auto_classes()
    register_auto_classes()
    assert isinstance(AutoConfig.from_pretrained(tmp_path), ArchCreditConfig)
    restored = AutoModel.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(restored(inputs)["logits"], expected)


@pytest.mark.parametrize("architecture", ["rnn", "transformer"])
@pytest.mark.parametrize("missing", ["architecture.readout.weight", "architecture.readout.bias"])
def test_missing_checkpoint_parameter_is_initialized(tmp_path, architecture, missing):
    from safetensors.torch import load_file, save_file

    model = ArchCreditModel(ArchCreditConfig(architecture=architecture)).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    checkpoint = tmp_path / "model.safetensors"
    saved = {name: value.clone() for name, value in load_file(checkpoint).items()}
    del saved[missing]
    save_file(saved, checkpoint, metadata={"format": "pt"})
    restored = ArchCreditModel.from_pretrained(tmp_path).eval()
    actual = restored.state_dict()
    for name, value in saved.items():
        torch.testing.assert_close(actual[name], value)
    assert torch.isfinite(actual[missing]).all()
    assert actual[missing].abs().max() < 1
    assert torch.isfinite(restored(torch.randn(2, 5, 2))["logits"]).all()
