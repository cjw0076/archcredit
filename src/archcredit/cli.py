"""Reproducible local benchmark runner: ``archbench run ...``."""

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import torch

from .benchmarks import generate_batch, normalize_task, task_spec
from .config import ModelConfig
from .credit import apply_gradients
from .diagnostics import CreditAlignment
from .loss import task_loss
from .registry import create_architecture, create_credit_rule


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.steps < 1 or args.lr <= 0 or not math.isfinite(args.lr):
        raise ValueError("steps must be positive and lr must be finite and positive")
    if args.plugin:
        # Console scripts start in their Scripts/bin directory. Make checkout
        # plugins importable just like python -m, then restore import search.
        previous_path = sys.path.copy()
        sys.path.insert(0, str(Path.cwd()))
        try:
            importlib.import_module(args.plugin)
        finally:
            sys.path[:] = previous_path
    task = normalize_task(args.task)
    spec = task_spec(task)
    config = ModelConfig(spec.input_size, args.hidden_size, spec.output_size, num_heads=1)
    train_rng = torch.Generator().manual_seed(args.seed)
    eval_rng = torch.Generator().manual_seed(args.seed + 1)
    started = time.perf_counter()
    alignment = CreditAlignment()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        model = create_architecture(args.model, config)
        rule = create_credit_rule(args.credit)
        reference = create_credit_rule("bptt")
        batch = generate_batch(task, args.batch_size, args.horizon, train_rng)
        compile_max_error = None
        if args.compile:
            compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
            with torch.no_grad():
                eager_output = model(batch.inputs)
                compiled_output = compiled(batch.inputs)
                torch.testing.assert_close(compiled_output, eager_output)
                compile_max_error = float((compiled_output - eager_output).abs().max())
        for step in range(args.steps):
            if step:
                batch = generate_batch(task, args.batch_size, args.horizon, train_rng)
            before_candidate_rng = torch.random.get_rng_state()
            gradients = rule.compute_gradients(model, batch)
            # Both directions come from identical parameters and the same batch.
            if step == args.steps - 1:
                after_candidate_rng = torch.random.get_rng_state()
                torch.random.set_rng_state(before_candidate_rng)
                try:
                    exact = reference.compute_gradients(model, batch)
                finally:
                    torch.random.set_rng_state(after_candidate_rng)
                alignment.update(
                    {name: -gradient for name, gradient in gradients.items()},
                    {name: -gradient for name, gradient in exact.items()},
                )
            apply_gradients(model, gradients, args.lr)
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ValueError("Non-finite parameters after update; reduce --lr")
        evaluation = generate_batch(task, args.batch_size, args.horizon, eval_rng)
        model.eval()
        with torch.no_grad():
            train_loss = float(task_loss(model(batch.inputs), batch))
            eval_loss = float(task_loss(model(evaluation.inputs), evaluation))
        if not math.isfinite(train_loss) or not math.isfinite(eval_loss):
            raise ValueError("Non-finite loss; reduce --lr")
        result: dict[str, object] = {
            "model": args.model,
            "credit": args.credit,
            "task": task,
            "seed": args.seed,
            "horizon": args.horizon,
            "sequence_length": batch.inputs.shape[1],
            "steps": args.steps,
            "batch_size": args.batch_size,
            "input_size": spec.input_size,
            "output_size": spec.output_size,
            "hidden_size": args.hidden_size,
            "learning_rate": args.lr,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "parameters": sum(p.numel() for p in model.parameters()),
            "rho": alignment.rho,
            "rho_scope": "final training batch before update; alignment, not task success",
            "compile_backend": "aot_eager" if args.compile else None,
            "compile_max_error": compile_max_error,
            "elapsed_seconds": time.perf_counter() - started,
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="archbench")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run", help="train and evaluate a synthetic delayed task")
    command.add_argument("--model", default="rnn")
    command.add_argument("--credit", default="bptt")
    command.add_argument("--task", default="delayed-xor")
    command.add_argument("--horizon", type=int, default=16)
    command.add_argument("--steps", type=int, default=20)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--hidden-size", type=int, default=16)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--lr", type=float, default=0.01)
    command.add_argument("--output", type=Path)
    command.add_argument(
        "--plugin", help="import a module that registers architectures or credit rules"
    )
    command.add_argument("--compile", action="store_true", help="aot_eager fullgraph forward smoke")
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, ImportError, NotImplementedError) as error:
        parser.error(str(error))
    document = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document + "\n", encoding="utf-8")
    print(document)


if __name__ == "__main__":
    main()
