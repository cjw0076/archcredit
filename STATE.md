# Project state

Updated: 2026-10-01 (Asia/Seoul).

## Current implementation

- Package name: `archcredit`; version: `0.1.0`.
- Public source baseline published at https://github.com/cjw0076/archcredit, commit `b7d9240`. Remote CI compatibility repair is underway; remote green status is not yet established.
- Architecture/CreditRule protocols, registries, serializable configuration, reference dense tanh RNN and causal Transformer are implemented.
- BPTT supports both references. Diagonal-local e-prop and Hebbian/Oja support the RNN and explicitly reject unsupported models.
- Delayed XOR, four-bit delayed copy, and adding use seeded generators and masked MSE.
- Alignment diagnostics, measured-horizon proxy scaffold, native checkpoints, optional Hugging Face local adapter, compile smoke, CLI, packaging, examples, and CI files are present.
- `cosmos` is a reserved failure-explicit placeholder. `cosmos_example` is a toy extension.
- README, contribution guide, Code of Conduct, MIT license, design boundaries, name-search evidence, release steps, and model card template are authored.

## Validation evidence available

- Documented editable dev installation completed successfully with default build isolation.
- Minimal training example completed successfully.
- RNN/BPTT delayed-XOR CLI completed 10 steps with finite outputs and reference alignment `rho = 1.0`.
- The example plugin completed through the module CLI; its zero update has undefined alignment, represented as JSON `null`.
- The installed console-script plugin import issue was fixed and retested without `PYTHONPATH`.
- Independent reviewer pass: APPROVE with no findings; targeted suite 80 passed, Ruff passed, and nonincremental mypy passed for 14 source files. This state records that separate pass; it does not self-approve the authoring lane.
- A 12-run synthetic smoke matrix covered all four supported model/credit pairs across all three tasks; all losses were finite. BPTT aligned with itself at `rho = 1.0` for both references. See [the verification report](docs/verification.md).
- Full integrated local suite: 104 passed in 132.15 seconds, zero skips. Ruff and mypy passed; native/Hugging Face serialization and compile smoke are included.
- Wheel and source distribution builds and Twine metadata checks passed with metadata version 2.4. Fresh wheel import/CLI checks passed outside the source checkout. Remote Actions evidence remains pending; rebuild the final artifacts after documentation updates.
- Final-wheel smoke confirmed imports from the installed wheel outside the checkout and a finite two-step e-prop/delayed-copy run. This smoke does not establish convergence.
- Initial hosted core jobs passed on Python 3.10, 3.11, and 3.12. The latest Transformers 5 Hugging Face job failed on adapter lifecycle compatibility; a fix and explicit 4.57.1/5.x CI coverage are in progress.

## Next steps

1. Complete the Hugging Face compatibility repair and collect passing remote GitHub Actions evidence, rebuilding final artifacts after documentation updates.
2. Add multi-seed learning curves, delay sweeps, baseline comparisons, and approximation ablations before claiming research quality.
3. After human release approval, stage TestPyPI and then PyPI with trusted publishing. For Hugging Face, train and evaluate a checkpoint, package remote custom code, fill the model card, and independently test loading before publication.

No PyPI or Hugging Face model release is recorded. Smoke tests do not establish convergence, real-data quality, paper reproduction, or upstream acceptance.
