"""Small, reproducible delayed tasks; ``horizon`` controls the blank delay."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Batch:
    inputs: Tensor
    targets: Tensor
    mask: Tensor


@dataclass(frozen=True)
class TaskSpec:
    input_size: int
    output_size: int


def normalize_task(task: str) -> str:
    task = task.replace("_", "-")
    if task == "copy":
        task = "delayed-copy"
    if task not in ("delayed-xor", "delayed-copy", "adding"):
        raise ValueError(f"Unknown task {task!r}; choose delayed-xor, delayed-copy, or adding")
    return task


def task_spec(task: str) -> TaskSpec:
    return TaskSpec(1 if normalize_task(task) == "delayed-xor" else 2, 1)


def generate_batch(
    task: str,
    batch_size: int,
    horizon: int,
    generator: torch.Generator,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Batch:
    """Return [B, T, channels] tensors with recall-only supervision.

    XOR observes two bits at t=0,1, then ``horizon`` blank steps and an answer.
    Copy observes four payload bits, ``horizon`` blank steps, a cue, and four
    recall steps. The recall input is zero, never the target. Adding marks
    two random numbers at t=0,1, then ``horizon`` distractor steps and an answer;
    subsequent unmarked random numbers are distractors. All sampling uses the
    supplied CPU generator and leaves global random state untouched.
    """
    task = normalize_task(task)
    if batch_size < 1 or horizon < 2:
        raise ValueError("batch_size must be positive and horizon must be at least 2")
    payload_length = 4
    length = horizon + (2 * payload_length + 1 if task == "delayed-copy" else 3)
    inputs = torch.zeros(batch_size, length, task_spec(task).input_size, dtype=dtype)
    targets = torch.zeros(batch_size, length, 1, dtype=dtype)
    mask = torch.zeros(batch_size, length, 1, dtype=torch.bool)
    mask[:, -1] = True
    if task == "delayed-xor":
        bits = torch.randint(2, (batch_size, 2), generator=generator)
        inputs[:, :2, 0] = bits.to(dtype)
        targets[:, -1, 0] = (bits[:, 0] != bits[:, 1]).to(dtype)
    elif task == "delayed-copy":
        payload = torch.randint(2, (batch_size, payload_length), generator=generator).to(dtype)
        inputs[:, :payload_length, 0] = payload
        inputs[:, -payload_length - 1, 1] = 1
        targets[:, -payload_length:, 0] = payload
        mask[:, -payload_length:] = True
    else:
        values = torch.rand(batch_size, length, generator=generator, dtype=dtype)
        inputs[:, :, 0] = values
        inputs[:, :2, 1] = 1
        targets[:, -1, 0] = values[:, :2].sum(dim=1)
    return Batch(inputs.to(device), targets.to(device), mask.to(device))
