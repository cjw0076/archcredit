# archcredit

**PyTorch-native open laboratory for neural architectures and credit assignment.**

archcredit separates the forward architecture from the learning rule. Build a reference model, choose how it receives credit, and compare updates on reproducible sequence tasks. v0.1 is a small research foundation: it ships runnable baselines and diagnostics, with no pretrained checkpoints or claims of benchmark superiority.

Repository: [cjw0076/archcredit](https://github.com/cjw0076/archcredit). See [the name check](docs/name-check.md) for the collision-search snapshot. PyPI publication remains a separate release step.

## Quick start

From a local checkout with Python 3.10 or newer:

```powershell
python -m pip install -e ".[dev]"
archbench run --model rnn --credit bptt --task delayed-xor --horizon 16 --steps 10 --seed 0
python examples/minimal.py
```

Use `--task delayed-copy` or `--task adding` for the other tasks. Use `--credit eprop` or `--credit hebbian` with the RNN to compare local updates. The CLI emits experiment configuration and measurements; a short smoke run does not establish that a task has been learned.

```powershell
archbench run --model transformer --credit bptt --task adding --horizon 16 --steps 2 --seed 0 --compile
archbench run --model toy_rnn --credit zero_credit --task delayed-xor --horizon 16 --steps 2 --plugin examples.plugin
```

## Core API

| API | Contract |
| --- | --- |
| `ModelConfig` | Serializable input, hidden, and output dimensions plus Transformer head/layer counts. |
| `Architecture` | A PyTorch module accepting `[batch, time, input_size]` and returning `[batch, time, output_size]`. |
| `CreditRule` | `compute_gradients(model, batch)` returns named parameter update tensors, in gradient sign convention. |
| `create_architecture(name, config)` | Creates a registered architecture. |
| `create_credit_rule(name)` | Creates a registered credit rule. |
| `Batch` | Contains `inputs`, `targets`, and a supervision `mask`. |
| `task_loss(predictions, batch)` | Masked mean squared error shared across learning rules. |
| `apply_gradients(model, gradients, lr)` | Applies `parameter -= lr * gradient`. |

See [the minimal training example](examples/minimal.py) for imports and a complete training step. The separate architecture and credit registries are the extension boundary; they do not imply that every rule supports every architecture.

## What works in v0.1

| Architecture | BPTT | e-prop approximation | Hebbian/Oja |
| --- | --- | --- | --- |
| Dense tanh RNN | Supported | Supported | Supported |
| Causal Transformer | Supported | Rejected | Rejected |
| Cosmos | Reserved placeholder | Unimplemented | Unimplemented |

**BPTT** differentiates the masked task loss through the full sequence. **e-prop** uses diagonal recurrent eligibility traces with symmetric readout feedback. It omits cross-unit recurrent paths, so it is an approximation for the dense RNN, not a reproduction of all variants in [the e-prop paper](https://arxiv.org/abs/1901.09049). **Hebbian** combines a local Oja update for hidden weights with a supervised readout update; the hidden bias uses a constant presynaptic feature of one. Its hidden updates are not necessarily loss gradients. Neither local rule silently falls back to BPTT.

The three synthetic tasks are delayed XOR, delayed copy, and adding. Each uses masked MSE, an explicit delay/horizon, and seeded generation. They exercise a long-range dependency; they are not evidence of success on real data. Compare multiple seeds, training budgets, and horizons before drawing conclusions.

| Task | Input and supervised answer | Sequence length |
| --- | --- | --- |
| `delayed-xor` | Two bits, blank delay, final XOR | `horizon + 3` |
| `delayed-copy` | Four payload bits, blank delay, recall cue, four supervised recall steps | `horizon + 9` |
| `adding` | Two marked random numbers, unmarked distractors, final sum | `horizon + 3` |

`--horizon` counts delay/distractor steps and must be at least 2. Targets are excluded from recall inputs.

## Diagnostics and compatibility

Credit alignment `rho` is the pooled cosine similarity of a rule's update direction and the BPTT update direction, evaluated on the same batch and parameters before applying an update. Zero-norm directions have undefined alignment. Positive alignment alone does not prove learning quality or biological plausibility.

The effective credit horizon utility is a scaffold: it reports the contiguous tested horizon range meeting an alignment threshold. It is an alignment proxy, not a validated measurement of causal credit reach or task memory. See [the design notes](docs/design.md).

`torch.compile` smoke checks use CPU `aot_eager` with full-graph forward/backward tests. CLI compilation covers the forward model. This does not establish GPU/Inductor support, speedups, or compilation of the local-rule trace computation. See [PyTorch's compile documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

Native checkpoints round-trip model configuration and state dictionaries using `weights_only` loading. The optional Hugging Face adapter wraps numeric sequence inputs, supports local `save_pretrained`/load and AutoClass registration, and has a serialization smoke test. Install the `hf` extra to use it. It is a scaffold: a standalone remote Hub model still needs its packaged custom code, trained weights, and model card. See [Hugging Face's custom model guide](https://huggingface.co/docs/transformers/custom_models).

## Contribute in five minutes

Copy [examples/plugin.py](examples/plugin.py), register a configuration-to-module factory or a zero-argument credit-rule factory, and pass the module to `--plugin`. A credit rule only needs to return tensors keyed by `model.named_parameters()`. State unsupported model/rule combinations explicitly.

The example's `zero_credit` rule is a no-learning control for testing registration and CLI wiring.

[CONTRIBUTING.md](CONTRIBUTING.md) explains the extension recipe and checks. [examples/cosmos_placeholder.py](examples/cosmos_placeholder.py) demonstrates a toy extension named `cosmos_example`; it does not implement the planned Cosmos architecture. The reserved `cosmos` registry entry raises `NotImplementedError` until its research contract and implementation are supplied.

## Development and release

```powershell
python -m ruff check .
python -m mypy src/archcredit
python -m pytest
python -m build
```

GitHub Actions runs lint, types, tests, compile smoke, and serialization checks. See [release steps](docs/releasing.md) for PyPI, Hugging Face, and upstream preparation. Package metadata and build configuration are provided; installation from PyPI requires a separately approved release.

Licensed under [MIT](LICENSE). Participation follows the [Code of Conduct](CODE_OF_CONDUCT.md).
