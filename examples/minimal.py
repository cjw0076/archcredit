"""Run with ``python -m examples.minimal`` after installing archcredit."""

import torch

from archcredit import ModelConfig, create_architecture, create_credit_rule
from archcredit.benchmarks import generate_batch
from archcredit.loss import task_loss


def main() -> None:
    torch.manual_seed(0)
    model = create_architecture("rnn", ModelConfig(input_size=1, hidden_size=8, output_size=1))
    rule = create_credit_rule("bptt")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    generator = torch.Generator().manual_seed(0)
    for _ in range(3):
        batch = generate_batch("delayed-xor", batch_size=4, horizon=8, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        gradients = rule.compute_gradients(model, batch)
        for name, parameter in model.named_parameters():
            parameter.grad = gradients[name]
        optimizer.step()
    print(f"Final batch loss: {task_loss(model(batch.inputs), batch).item():.6f}")


if __name__ == "__main__":
    main()
