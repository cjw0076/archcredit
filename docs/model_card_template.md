# Model card template

Replace every placeholder before publishing. This template is not a trained model card.

## Identity and intended use

- Model name and immutable revision: `<fill in>`
- Architecture and configuration: `<fill in>`
- Credit rule and approximations: `<fill in>`
- License and maintainer: `<fill in>`
- Intended research use: `<fill in>`
- Unsupported uses and inputs: `<fill in>`

## Training

- Source commit and environment: `<fill in>`
- Dataset or task generator, version, and split policy: `<fill in>`
- Horizon, seeds, batches, optimizer/update rule, learning rate, and budget: `<fill in>`
- Compute and checkpoint selection policy: `<fill in>`
- Reproduction instructions verified in a fresh environment: `<fill in>`

## Evaluation

Report task metrics, learning curves, multiple seeds, uncertainty, baselines, and failure cases. State whether evaluation uses synthetic tasks or external data. For alignment, record reference BPTT parameters/batch, parameter coverage, and zero-norm handling. Label the horizon scaffold as an alignment proxy.

## Loading and limitations

Document package versions, numeric input shapes, custom-code requirements, immutable revision pins, and the tested save/load path. Include a verified inference example when a checkpoint exists. State untested devices/backends, known failure modes, and differences from any cited paper.

## Provenance

List training/evaluation artifacts, checkpoint hashes, approvals for release, and independent reproduction evidence. Do not substitute smoke tests or random-weight serialization for trained-model quality.
