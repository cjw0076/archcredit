import pytest
import torch

from archcredit.benchmarks import generate_batch, task_spec
from archcredit.loss import task_loss


@pytest.mark.parametrize("task", ["delayed-xor", "delayed-copy", "adding"])
def test_deterministic_and_global_rng_untouched(task):
    state = torch.random.get_rng_state().clone()
    first = generate_batch(task, 8, 5, torch.Generator().manual_seed(3))
    second = generate_batch(task, 8, 5, torch.Generator().manual_seed(3))
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.targets, second.targets)
    length = 14 if task == "delayed-copy" else 8
    assert first.inputs.shape == (8, length, task_spec(task).input_size)
    assert first.targets.shape == (8, length, 1)
    assert first.mask.sum() == (32 if task == "delayed-copy" else 8)
    assert first.mask[:, -1].all()
    prediction = torch.zeros_like(first.targets)
    expected = task_loss(prediction, first)
    prediction.masked_fill_(~first.mask, 999)
    assert task_loss(prediction, first) == expected


def test_xor_truth_table_and_delayed_inputs():
    batch = generate_batch("delayed-xor", 128, 4, torch.Generator().manual_seed(0))
    assert torch.equal(
        batch.targets[:, -1, 0], (batch.inputs[:, 0, 0] != batch.inputs[:, 1, 0]).float()
    )
    assert batch.inputs[:, 2:].count_nonzero() == 0
    assert torch.unique(batch.inputs[:, :2, 0], dim=0).shape[0] == 4


def test_copy_cue_and_no_recall_leakage():
    batch = generate_batch("copy", 32, 4, torch.Generator().manual_seed(0))
    assert torch.equal(batch.targets[:, -4:, 0], batch.inputs[:, :4, 0])
    assert batch.inputs[:, -4:].count_nonzero() == 0
    assert batch.inputs[:, 4:-5].count_nonzero() == 0
    assert batch.inputs[:, -5, 1].eq(1).all()
    assert batch.inputs[:, -5, 0].eq(0).all()
    assert batch.targets[:, -4:].count_nonzero() > 0


def test_adding_target_only_uses_marked_early_values():
    batch = generate_batch("adding", 16, 4, torch.Generator().manual_seed(2))
    marked_sum = (batch.inputs[:, :, 0] * batch.inputs[:, :, 1]).sum(dim=1)
    torch.testing.assert_close(batch.targets[:, -1, 0], marked_sum)
    assert batch.inputs[:, 2:, 1].count_nonzero() == 0
    assert batch.targets[:, :-1].count_nonzero() == 0


@pytest.mark.parametrize(
    "task,batch_size,horizon", [("missing", 2, 2), ("copy", 0, 2), ("adding", 2, 1)]
)
def test_invalid_task_arguments(task, batch_size, horizon):
    with pytest.raises(ValueError):
        generate_batch(task, batch_size, horizon, torch.Generator())
