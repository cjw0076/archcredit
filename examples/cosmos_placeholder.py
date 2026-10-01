"""Extension example only: NOT an implementation or validation of Cosmos.

This tiny residual state update demonstrates the architecture boundary. Replace
it with a specification-backed implementation and scientific tests before making
any Cosmos research claim. The reserved core ``cosmos`` entry remains unavailable.
"""

import torch
from torch import Tensor, nn

from archcredit import ModelConfig, register_architecture


@register_architecture("cosmos_example")
class CosmosExample(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input = nn.Linear(config.input_size, config.hidden_size)
        self.interaction = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.readout = nn.Linear(config.hidden_size, config.output_size)
        self.hidden_size = config.hidden_size

    def forward(self, inputs: Tensor) -> Tensor:
        state = inputs.new_zeros(inputs.shape[0], self.hidden_size)
        outputs = []
        for frame in inputs.unbind(dim=1):
            state = 0.5 * state + 0.5 * torch.tanh(self.input(frame) + self.interaction(state))
            outputs.append(self.readout(state))
        return torch.stack(outputs, dim=1)
