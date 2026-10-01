# Release preparation

## Current distribution boundary

v0.1 includes package metadata, source/wheel build configuration, local examples, and CI. These artifacts do not imply that a PyPI release, pretrained model, or accepted upstream integration exists.

The name `archcredit` was selected after checks on 2026-10-01: its PyPI JSON endpoint returned 404 and GitHub exact name search found no repositories. `neurocredit` and `creditlab` also had no PyPI project at the time, but had GitHub name collisions associated with financial projects. Availability checks are a snapshot, not a reservation or trademark clearance; recheck immediately before release.

## Python package

1. Require green lint, types, tests, compile, and native/Hugging Face serialization checks, plus independent review.
2. Build wheel and source distribution and install the wheel into a fresh environment. Run the CLI and minimal example from that installation.
3. Verify version, license, README rendering, included files, dependency bounds, and repository URLs.
4. Configure PyPI trusted publishing for the actual repository and release environment. Stage a TestPyPI release and test installation independently.
5. Obtain explicit human approval before publishing to PyPI. Publish the reviewed artifacts, record hashes and provenance, and tag the corresponding commit.

This repository does not automate a package upload by default. Do not interpret a successful build as a published release.

## Hugging Face

1. Train a reproducible checkpoint and retain exact configuration, seeds, data-generation settings, environment, and learning curves.
2. Complete [the model card template](model_card_template.md), including limitations and honest diagnostic scope.
3. Package the adapter's custom configuration/model code and dependencies according to [the custom model documentation](https://huggingface.co/docs/transformers/custom_models). Local AutoClass registration is separate from remote custom-code packaging.
4. Save configuration and safetensors weights, then test loading from a fresh process/environment. For remote loading, explicitly document custom-code trust and version pins.
5. Obtain human approval for Hub publication and verify the uploaded revision independently. Report random-weight scaffolds and trained checkpoints distinctly.

## Upstream contribution

Choose a concrete reusable integration after external users can reproduce it: an adapter, benchmark, diagnostic, or architecture with validated weights. First establish correctness and a clear maintenance boundary, then discuss requirements with the target project's maintainers. A local custom model or repository publication does not establish acceptance into Transformers or another upstream project.

Prioritize contributor feedback on the registration recipe, capability errors, and example clarity before expanding architecture count.
