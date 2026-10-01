# Project state

Updated: 2026-10-01 (Asia/Seoul).

## v0.1 implementation

- Package: `archcredit`, version `0.1.0`; public source: https://github.com/cjw0076/archcredit.
- Architecture/CreditRule protocols, registries, serializable configuration, dense tanh RNN, and causal Transformer are implemented.
- BPTT supports both references. Diagonal-local e-prop and Hebbian/Oja support the RNN and explicitly reject unsupported models.
- Seeded delayed XOR, four-bit delayed copy, and adding tasks use masked MSE.
- Alignment diagnostics, measured-horizon proxy scaffold, native checkpoints, CPU compile smoke, CLI, packaging, contributor examples, and CI are implemented.
- The optional Hugging Face adapter supports local configuration/safetensors/AutoClass loading with tested complete/partial checkpoint initialization on Transformers 4.57.1 and 5.18. Dependency range: `>=4.57.1,<6`.
- `cosmos` is reserved and fails explicitly. `cosmos_example` is a toy extension, not the planned Cosmos architecture.
- README, contribution guide, Code of Conduct, MIT license, design boundaries, name-search evidence, release guide, model card template, and verification report are provided.

## Verification

- Final local suite: **108 passed in 56.48 seconds, zero skips**. Ruff passed; nonincremental mypy passed for 14 source files.
- Independent source review and final narrow Hugging Face review returned APPROVE with no findings. These were separate passes from authoring.
- Source revision `f1327816fc749761a55f8642d4f61e194f560bed` passed all five hosted jobs in [Actions run 36816992319](https://github.com/cjw0076/archcredit/actions/runs/36816992319): core Python 3.10/3.11/3.12 each passed 102 tests with one optional-HF module skip; Transformers 4.57.1/5.18 each passed 6 HF tests.
- Final wheel/source builds and both Twine checks passed. Outside-checkout wheel import/CLI smoke passed; installed contributor plugin works without `PYTHONPATH`.
- Twelve synthetic smoke runs covered the four supported model/credit pairs and three tasks with finite losses. BPTT alignment was `rho = 1.0`; zero-credit alignment was JSON `null`.
- See [docs/verification.md](docs/verification.md) for runtime, source receipt, and evidence limits. The hosted receipt identifies its tested source revision; it does not claim a future documentation commit was tested.

## Next work

1. Establish research quality with multi-seed learning curves, delay sweeps, baseline comparisons, approximation ablations, and failure cases.
2. After human release approval, stage TestPyPI and publish to PyPI using trusted publishing.
3. Train and evaluate real Hugging Face checkpoints, complete model cards and remote custom-code packaging, and independently reproduce loading/results before Hub publication or upstream proposals.

No PyPI or Hugging Face model release is recorded. The effective horizon remains an alignment scaffold; CPU `aot_eager` smoke does not establish Inductor/GPU performance or compilation of online local learning. Passing tests do not establish convergence, real-data quality, paper reproduction, or upstream acceptance.
