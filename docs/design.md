# v0.1 design and research boundaries

## Goal

A PyTorch-native laboratory should let contributors change forward computation and credit assignment independently, with minimal glue. v0.1 therefore uses two registries, a serializable model configuration, sequence batches, and named parameter updates. Ordinary PyTorch modules remain the computation boundary.

The reference design is the retrieved conversation “오픈소스 기여 설계,” proposing separation of forward architecture, credit architecture, and state/memory. The specifically requested uploaded `/mnt/data/오픈소스 기여 설계.txt` was not accessible in this environment. The retrieved conversation supplies the available design context; this implementation must not claim byte-for-byte validation against that upload.

## Update contract

`compute_gradients` returns update tensors without applying them. This permits alignment measurement before mutation. `apply_gradients` subtracts learning-rate-scaled tensors, so local learning rules express their desired direction using the same sign convention as a loss gradient. This API name is a convention: a Hebbian tensor need not be a mathematical gradient of the task loss.

BPTT supports both reference architectures. Local rules require the dense tanh RNN's exposed recurrence and readout structure. The Transformer rejects them explicitly. General credit/architecture interoperability is future work.

## Local credit approximation

For a hidden unit `j` and presynaptic coordinate `i`, the e-prop implementation carries a diagonal eligibility trace:

```text
E[t,j,i] = (1 - h[t,j]^2) * (pre[t,i] + Wrec[j,j] * E[t-1,j,i])
```

The learning signal uses the masked output derivative projected through the readout weights. Only self-recurrent eligibility transport is retained; cross-unit paths are omitted. This can coincide with full recurrent differentiation for a scalar hidden recurrence, but it is an approximation for a general dense recurrence. It is inspired by [Bellec et al.'s e-prop work](https://arxiv.org/abs/1901.09049), not a claim of complete paper reproduction.

Hebbian learning uses an Oja-style hidden-weight rule and a supervised readout. The hidden bias uses the same local rule with a constant presynaptic feature of one. Its behavior should be evaluated as a local update baseline, not presumed to optimize the masked task loss.

## Diagnostics

Alignment pools matching parameter tensors into one cosine similarity. Compare rule and BPTT directions on the same batch and parameters, before either update. Multiplying both gradient vectors by minus one leaves their cosine unchanged. A zero norm makes alignment undefined; do not convert that case into successful alignment.

The effective credit horizon scaffold evaluates alignment across supplied horizons and finds a contiguous threshold-satisfying range. The threshold and sampled horizons are experiment choices. This proxy does not establish memory capacity, causal influence, task success, or a physical propagation distance. Validate those separately with delay sweeps, baselines, and controlled interventions.

## Experimental extensions

`cosmos` is reserved and fails explicitly. The independent `cosmos_example` plugin demonstrates registration with a toy computation. A flagship Cosmos implementation needs its own state update, interaction topology, credit rule, reset semantics, and falsifiable experiments.

Hugging Face integration is optional. Numeric sequences do not imply tokenizer support or language-model generation. Local configuration/weight serialization is the initial adapter boundary. Hub distribution and independent remote loading require additional release validation.

Compile evidence covers CPU `aot_eager` full-graph reference forward/backward smoke. It does not cover all plugins, the local credit routines, GPU kernels, Inductor, or performance. Follow [PyTorch's compile API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) when extending compatibility.
