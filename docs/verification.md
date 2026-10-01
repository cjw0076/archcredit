# v0.1 verification evidence

Recorded on 2026-10-01. This report distinguishes execution checks from learning-quality evidence.

## Environment

- Windows 11; Python 3.11.9.
- PyTorch 2.14.0+cpu; Transformers 4.57.1.
- CPU execution. GPU, Inductor, and other dependency-version combinations were not established by this local run.

## Checks

| Check | Evidence |
| --- | --- |
| Independent review | Separate reviewer returned APPROVE with no findings. |
| Reviewer targeted tests | 80 passed. |
| Ruff | Passed. |
| Type checking | Nonincremental mypy passed for 14 source files. |
| Full integrated test suite | 104 passed in 132.15 seconds; zero skips. |
| Contributor examples | Minimal example and installed console-script plugin command passed; plugin retest did not require `PYTHONPATH`. |
| Compile | CPU `aot_eager` full-graph reference forward/backward smoke; no performance claim. |
| Native serialization | Configuration/state-dictionary round-trip exercised by serialization tests. |
| Hugging Face | Local configuration/safetensors/AutoClass scaffold exercised; no trained checkpoint or remote Hub release claimed. |
| Distribution | Wheel/source builds and Twine checks passed; metadata version 2.4. Fresh wheel import/CLI passed outside the source checkout. Final artifacts must include the completed documentation. |
| GitHub | Public repository created at [cjw0076/archcredit](https://github.com/cjw0076/archcredit); final push and remote Actions result pending. |

The authoring pass and independent review ran in separate lanes. Local checks do not substitute for remote CI or release approval.

## Synthetic smoke matrix

[artifacts/smoke-matrix.json](../artifacts/smoke-matrix.json) records 12 runs: RNN/BPTT, RNN/e-prop, RNN/Hebbian, and Transformer/BPTT, each on delayed XOR, delayed copy, and adding. Each run used horizon 8, 5 update steps, batch size 8, and seed 7.

All runs produced finite losses. BPTT reference alignment was `rho = 1.0` for both reference architectures. The recorded e-prop/Hebbian alignment and losses are short-run observations, not convergence, task mastery, superiority, or paper-reproduction evidence. The zero-credit plugin's undefined alignment is represented by JSON `null`.

## Limits and next evidence

The effective credit horizon is an alignment-based scaffold across measured delays. It is not a validated measure of causal credit reach or task memory. Compile checks cover reference forward/backward graph capture with `aot_eager`; they do not establish Inductor/GPU support or compilation of online local-rule state updates.

Before research claims, add learning curves, multiple seeds, delay sweeps, baseline comparisons, approximation ablations, and failure cases. Before release claims, collect remote CI, approved PyPI publication, and independently loaded Hub artifacts. Cosmos remains a reserved extension and a separate toy example, not a completed flagship architecture.
