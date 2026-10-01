"""Two small contributor extensions loaded with ``--plugin examples.plugin``.

zero_credit is a no-learning control, not a scientifically useful training rule.
"""

import torch
from torch import Tensor, nn

from archcredit import ModelConfig, register_architecture, register_credit_rule
from archcredit.benchmarks import Batch


@register_architecture("toy_rnn")
class ToyRNN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.cell = nn.Linear(config.input_size + config.hidden_size, config.hidden_size)
        self.readout = nn.Linear(config.hidden_size, config.output_size)
        self.hidden_size = config.hidden_size

    def forward(self, inputs: Tensor) -> Tensor:
        state = inputs.new_zeros(inputs.shape[0], self.hidden_size)
        outputs = []
        for frame in inputs.unbind(dim=1):
            state = torch.tanh(self.cell(torch.cat((frame, state), dim=-1)))
            outputs.append(self.readout(state))
        return torch.stack(outputs, dim=1)


@register_credit_rule("zero_credit")
class ZeroCredit:
    def compute_gradients(self, model: nn.Module, batch: Batch) -> dict[str, Tensor]:
        return {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters()}
