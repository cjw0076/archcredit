# v0.1 verification evidence

Recorded on 2026-10-01. v0.1 implementation checks pass locally and on the hosted source revision below. These checks establish executable behavior, not learning quality.

## Runtime and source receipt

- Local: Windows 11, Python 3.11.9, PyTorch 2.14.0+cpu, Transformers 4.57.1.
- Hosted core: Python 3.10, 3.11, and 3.12. Hosted HF: Transformers 4.57.1 and 5.18.
- Source: [`f1327816fc749761a55f8642d4f61e194f560bed`](https://github.com/cjw0076/archcredit/commit/f1327816fc749761a55f8642d4f61e194f560bed).
- Hosted receipt: [Actions run 36816992319](https://github.com/cjw0076/archcredit/actions/runs/36816992319), **all five jobs successful**.

This receipt identifies the tested source snapshot. It does not claim coverage of a future documentation commit, a PyPI release, or a Hub model publication.

## Final checks

| Check | Result |
| --- | --- |
| Local full suite | 108 passed in 56.48 seconds; zero skips. |
| Ruff | Passed. |
| Types | Nonincremental mypy passed for 14 source files. |
| Hosted core matrix | Python 3.10/3.11/3.12 each passed 102 tests, with one optional-HF module skip. |
| Hosted HF matrix | Transformers 4.57.1 and 5.18 each passed 6 tests. |
| Independent review | Source review and final narrow HF review returned APPROVE with no findings, in separate lanes from authoring. |
| Contributor examples | Minimal example and installed console-script plugin passed; the plugin required no `PYTHONPATH`. |
| Compile | CPU `aot_eager` full-graph reference forward/backward parity passed. |
| Native serialization | Configuration/state-dictionary checkpoint round-trip passed. |
| Hugging Face | Local configuration/safetensors/AutoClass tests passed, including complete/partial checkpoint initialization regressions. |
| Distribution | Final wheel/source builds and both Twine checks passed; metadata version 2.4. Outside-checkout installed-wheel import/CLI smoke passed. |

The optional HF dependency range is `transformers>=4.57.1,<6`. Explicit model lifecycle initialization and normal initialization of missing checkpoint weights are covered by regression tests. The adapter remains a numeric-sequence scaffold with no trained checkpoint or remote Hub release claimed.

## Synthetic smoke matrix

[artifacts/smoke-matrix.json](../artifacts/smoke-matrix.json) records 12 runs: RNN/BPTT, RNN/e-prop, RNN/Hebbian, and Transformer/BPTT, each on delayed XOR, delayed copy, and adding. Each used horizon 8, 5 update steps, batch size 8, and seed 7.

All runs produced finite losses. BPTT reference alignment was `rho = 1.0` for both reference architectures. The e-prop/Hebbian alignment and losses are short-run observations, not convergence, task mastery, superiority, or paper-reproduction evidence. The zero-credit plugin's undefined alignment is JSON `null`. A separate outside-checkout wheel smoke completed a finite two-step e-prop/delayed-copy run; this likewise establishes execution only.

## Limits and next evidence

The effective credit horizon is an alignment scaffold across measured delays, not a validated measure of causal credit reach or memory. Compile checks cover reference forward/backward capture with CPU `aot_eager`; they do not establish Inductor/GPU performance or compilation of online local-rule updates. Cosmos remains a reserved extension and separate toy example.

Next research work is multi-seed learning curves, delay sweeps, baseline comparisons, approximation ablations, and failure analysis. Next releases require approved TestPyPI/PyPI publication and real trained HF checkpoints with completed model cards and remote custom code. Independently reproduce results and loading before making upstream integration claims.
