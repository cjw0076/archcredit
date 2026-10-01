"""Exact BPTT and explicitly approximate, graph-free local learning rules.

Diagonal e-prop retains only each neuron's recurrent self-connection in its
eligibility recurrence. It is exact for a scalar tanh RNN, but drops cross-neuron
paths in larger RNNs. This is an educational approximation, not the complete
spiking e-prop algorithm of Bellec et al. (https://arxiv.org/abs/1901.09049).
Hebbian hidden updates are Oja pseudo-gradients, not derivatives of task loss.
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .architectures import RNNArchitecture
from .loss import _time_mask, task_loss
from .registry import register_credit_rule

if TYPE_CHECKING:
    from .benchmarks import Batch


@register_credit_rule("bptt")
class BPTT:
    def compute_gradients(self, model: nn.Module, batch: "Batch") -> dict[str, Tensor]:
        parameters = dict(model.named_parameters())
        result = {name: torch.zeros_like(value) for name, value in parameters.items()}
        trainable = {name: value for name, value in parameters.items() if value.requires_grad}
        if not trainable:
            return result
        loss = task_loss(model(batch.inputs), batch)
        if not loss.requires_grad:
            return result
        gradients = torch.autograd.grad(loss, tuple(trainable.values()), allow_unused=True)
        for name, gradient in zip(trainable, gradients, strict=True):
            if gradient is not None:
                result[name] = gradient.detach()
        return result


def _readout_gradients(
    model: RNNArchitecture, states: Tensor, derivative: Tensor
) -> dict[str, Tensor]:
    return {
        "readout.weight": torch.einsum("bto,bth->oh", derivative, states),
        "readout.bias": derivative.sum(dim=(0, 1)),
    }


def _output_derivative(prediction: Tensor, batch: "Batch") -> Tensor:
    mask = _time_mask(prediction, batch)
    return (
        2 * (prediction - batch.targets) * mask / (mask.sum().clamp_min(1) * prediction.shape[-1])
    )


@register_credit_rule("eprop")
class EProp:
    """Diagonal-local eligibility traces with symmetric readout feedback.

    e[t,j,i] = (1-h[t,j]^2) * (pre[t,i] + Wrec[j,j]*e[t-1,j,i]).
    Hidden task pseudo-gradients sum (dL/dy @ Wout)[t,j] * e[t,j,i].
    There is no autograd graph or BPTT fallback.
    """

    @torch.no_grad()
    def compute_gradients(self, model: nn.Module, batch: "Batch") -> dict[str, Tensor]:
        if not isinstance(model, RNNArchitecture):
            raise ValueError("eprop supports only RNNArchitecture (diagonal-local approximation)")
        prediction, states, previous = model.forward_with_trace(batch.inputs)
        derivative = _output_derivative(prediction, batch)
        learning_signal = derivative @ model.readout.weight
        gradients = {name: torch.zeros_like(value) for name, value in model.named_parameters()}
        gradients.update(_readout_gradients(model, states, derivative))
        size = batch.inputs.shape[0]
        input_trace = batch.inputs.new_zeros(
            size, model.config.hidden_size, model.config.input_size
        )
        recurrent_trace = batch.inputs.new_zeros(
            size, model.config.hidden_size, model.config.hidden_size
        )
        bias_trace = batch.inputs.new_zeros(size, model.config.hidden_size)
        diagonal = model.recurrent.weight.diagonal()
        for time in range(batch.inputs.shape[1]):
            activation_derivative = 1 - states[:, time].square()
            input_trace = activation_derivative[:, :, None] * (
                batch.inputs[:, time, None, :] + diagonal[None, :, None] * input_trace
            )
            recurrent_trace = activation_derivative[:, :, None] * (
                previous[:, time, None, :] + diagonal[None, :, None] * recurrent_trace
            )
            bias_trace = activation_derivative * (1 + diagonal[None, :] * bias_trace)
            signal = learning_signal[:, time]
            gradients["input.weight"] += (signal[:, :, None] * input_trace).sum(dim=0)
            gradients["recurrent.weight"] += (signal[:, :, None] * recurrent_trace).sum(dim=0)
            gradients["input.bias"] += (signal * bias_trace).sum(dim=0)
        return gradients


@register_credit_rule("hebbian")
class Hebbian:
    """Oja hidden updates plus a supervised readout.

    Hidden delta W[j,i] = mean(h[j]*pre[i] - h[j]^2*W[j,i]);
    bias uses a constant presynaptic feature of one. Returned hidden
    pseudo-gradients are -delta W, so ordinary SGD applies the local update.
    These hidden updates ignore labels and do not claim long-range task credit.
    """

    @torch.no_grad()
    def compute_gradients(self, model: nn.Module, batch: "Batch") -> dict[str, Tensor]:
        if not isinstance(model, RNNArchitecture):
            raise ValueError("hebbian supports only RNNArchitecture (Oja local updates)")
        prediction, states, previous = model.forward_with_trace(batch.inputs)
        mean_square = states.square().mean(dim=(0, 1))
        count = states.shape[0] * states.shape[1]
        gradients = {
            "input.weight": -(
                torch.einsum("bth,bti->hi", states, batch.inputs) / count
                - mean_square[:, None] * model.input.weight
            ),
            "recurrent.weight": -(
                torch.einsum("bth,bti->hi", states, previous) / count
                - mean_square[:, None] * model.recurrent.weight
            ),
            "input.bias": -(states.mean(dim=(0, 1)) - mean_square * model.input.bias),
        }
        gradients.update(_readout_gradients(model, states, _output_derivative(prediction, batch)))
        return gradients


@torch.no_grad()
def apply_gradients(model: nn.Module, gradients: dict[str, Tensor], lr: float) -> None:
    """Validate the entire SGD update before mutation; preserve frozen parameters."""
    import math

    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    parameters = dict(model.named_parameters())
    if gradients.keys() != parameters.keys():
        raise ValueError("Gradient keys must exactly match model.named_parameters()")
    for name, parameter in parameters.items():
        gradient = gradients[name]
        if gradient.shape != parameter.shape:
            raise ValueError(f"Gradient shape mismatch for {name}")
        if gradient.device != parameter.device or gradient.dtype != parameter.dtype:
            raise ValueError(f"Gradient device/dtype mismatch for {name}")
        if not torch.isfinite(gradient).all():
            raise ValueError(f"Non-finite gradient for {name}")
        if parameter.requires_grad and not torch.isfinite(parameter - lr * gradient).all():
            raise ValueError(f"Non-finite proposed update for {name}")
    for name, parameter in parameters.items():
        if parameter.requires_grad:
            parameter.add_(gradients[name], alpha=-lr)
