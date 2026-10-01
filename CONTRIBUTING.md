# Contributing

Keep the extension boundary small. Prefer one runnable architecture or learning rule, an explicit capability limit, and a test over a new framework abstraction.

## Set up

Use Python 3.10 or newer and run from the checkout root:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

## Add an architecture or credit rule

1. Copy `examples/plugin.py` into an importable module.
2. For an architecture, decorate a factory with `register_architecture("your_name")`. The factory accepts `ModelConfig` and returns a PyTorch module with sequence-shaped inputs and outputs.
3. For a credit rule, decorate a zero-argument factory with `register_credit_rule("your_rule")`. Implement `compute_gradients(model, batch)` and return a dictionary of parameter names to tensors. Use gradient sign: `apply_gradients` subtracts the returned tensors times the learning rate.
4. Reject unsupported model types with a clear error. Do not substitute BPTT while claiming a different algorithm.
5. Run your module through `--plugin` and add a focused test for its shapes, finite updates, and capability limits.

The shipped plugin is runnable:

```powershell
archbench run --model toy_rnn --credit zero_credit --task delayed-xor --horizon 16 --steps 2 --plugin examples.plugin
```

Use `examples/minimal.py` when a custom training loop is more appropriate. Registration loads Python code, so use plugin modules you trust.

## Evidence for a change

For a new learning rule, describe the equations, approximation, state carried between steps, and where the implementation differs from its reference. Verify parameter-name/shape coverage and compare against BPTT on the same pre-update model when reporting alignment. Include a zero-direction case for diagnostics.

For a new benchmark, specify the dependency length, supervision mask, target generation, seeded randomness, and loss normalization. Report learning curves and multiple seeds before claiming improved credit assignment. Test losses decreasing only when the test setup supports a reproducible assertion; a finite update is a smoke check, not convergence evidence.

Cosmos is an extension point, not a completed architecture. Replace its placeholder only with an explicit forward/state/credit contract, executable implementation, and evidence of behavior. Do not relabel the toy `cosmos_example` as Cosmos.

## Check a pull request

```powershell
python -m ruff check .
python -m mypy src/archcredit
python -m pytest
python -m build
```

Keep examples and documentation consistent with the public API. State optional dependencies and untested hardware. A separate reviewer evaluates the change after the authoring pass; green checks alone are not reviewer approval.

Use an issue or pull request to explain the problem, final behavior, and validation. Follow [the Code of Conduct](CODE_OF_CONDUCT.md).
