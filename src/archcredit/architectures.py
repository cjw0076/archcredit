"""Transparent reference models returning one prediction per time step."""

import math

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .registry import register_architecture


@register_architecture("rnn")
class RNNArchitecture(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.architecture_name = "rnn"
        self.input = nn.Linear(config.input_size, config.hidden_size)
        self.recurrent = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.readout = nn.Linear(config.hidden_size, config.output_size)

    def forward_with_trace(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = inputs.new_zeros(inputs.shape[0], self.config.hidden_size)
        states: list[Tensor] = []
        previous_states: list[Tensor] = []
        for time in range(inputs.shape[1]):
            previous_states.append(hidden)
            hidden = torch.tanh(self.input(inputs[:, time]) + self.recurrent(hidden))
            states.append(hidden)
        trace = torch.stack(states, dim=1)
        return self.readout(trace), trace, torch.stack(previous_states, dim=1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_with_trace(inputs)[0]


@register_architecture("transformer")
class TransformerArchitecture(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise ValueError("Transformer hidden_size must be divisible by num_heads")
        self.config = config
        self.architecture_name = "transformer"
        self.input = nn.Linear(config.input_size, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=2 * config.hidden_size,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.num_layers, enable_nested_tensor=False)
        self.readout = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        length = inputs.shape[1]
        position = torch.arange(length, device=inputs.device, dtype=inputs.dtype)[:, None]
        channel = torch.arange(self.config.hidden_size, device=inputs.device)
        frequency = torch.exp(
            -math.log(10000.0) * (2 * (channel // 2)).to(inputs.dtype) / self.config.hidden_size
        )
        phase = position * frequency[None, :]
        encoding = torch.where(channel[None, :] % 2 == 0, phase.sin(), phase.cos())
        hidden = self.input(inputs) + encoding[None, :, :]
        causal_mask = torch.ones(length, length, device=inputs.device, dtype=torch.bool).triu(1)
        return self.readout(self.encoder(hidden, mask=causal_mask, is_causal=True))


@register_architecture("cosmos")
def cosmos_placeholder(config: ModelConfig) -> nn.Module:
    raise NotImplementedError(
        "Cosmos is an experimental extension point, not an implemented model. "
        "Register your implementation under a new name; see examples/cosmos_placeholder.py."
    )
