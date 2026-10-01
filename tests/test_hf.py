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
    model = ArchCreditModel(config).eval()
    inputs = torch.randn(2, 5, 2)
    expected = model(inputs)["logits"].detach()
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / "model.safetensors").is_file()
    register_auto_classes()
    register_auto_classes()
    assert isinstance(AutoConfig.from_pretrained(tmp_path), ArchCreditConfig)
    restored = AutoModel.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(restored(inputs)["logits"], expected)
