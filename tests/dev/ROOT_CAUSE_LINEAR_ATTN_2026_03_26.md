# Root Cause Finding (2026-03-26)

## Conclusion

The primary root cause of lower CoreML/ANE accuracy vs HF fp16 is:

- Small ANE numerical error in linear-attention recurrence (`_chunk_gated_delta_rule`)
- Amplification of that perturbation in the subsequent `RMSNormGated + out_proj` chain
- Resulting large channel-0 bias and early argmax flips during decode

Dynamic KV cache writes are functioning correctly and are not the dominant root cause for the observed HF parity drop.

## Evidence From Fresh Runs

### 1) Single-layer isolation (`tests/dev/_debug_single_layer_parity.py`)

- Linear-attention full layer (layer 0): cosine = 0.9904965845
- Linear-attention-only path: cosine = 0.9940010003
- Full-attention full layer (layer 3): cosine = 0.9999606629
- MLP-only path: cosine = 0.9997851083

Interpretation:
- Divergence is concentrated in linear-attention path, not full-attention or MLP.

### 2) CoreNorm decomposition (`tests/dev/_debug_corenorm_decomp.py`)

- Recurrence-only: cosine = 0.9999430550, max_abs = 0.002930
- Norm+Projection (with PyTorch recurrence input): cosine = 0.9999499707
- L2 norm isolate: cosine = 1.0000000000
- Full CoreNorm reference from same diagnostic family: about 0.9956

Interpretation:
- Each subcomponent alone is near-perfect.
- Combined path still drops to about 0.9956, indicating amplification of small recurrence perturbations.

### 3) Split-CoreNorm falsification tests (`tests/dev/_test_split_corenorm.py`)

- Baseline combined CoreNorm: cosine = 0.9955588588
- Remove fuse/merge MIL passes: cosine = 0.9955588588 (no improvement)
- Separate-model cascade (recurrence then norm+proj): cosine = 0.9955588588 (no improvement)
- Empty pipeline and state-barrier approaches failed to deploy/load on ANE in this run

Interpretation:
- Not caused by pass-level fusion in a way removable by standard pass toggles.
- Splitting models does not remove the degradation.

### 4) Direct amplification proof (`tests/dev/_diag_normprojonly.py`)

- Recurrence output parity (PyTorch vs ANE): cosine = 0.9999430550, max diff = 0.002930
- Norm+proj fed PyTorch recurrence: cosine = 0.9999499707
- Norm+proj fed ANE recurrence: cosine = 0.9955588588
- Difference between those two norm+proj outputs: cosine = 0.9954888540, max_diff = 0.292969

Interpretation:
- The same norm+proj model behaves well with PyTorch recurrence input.
- Quality drops when input recurrence tensor comes from ANE recurrence path.
- This is direct evidence of perturbation amplification.

## Supporting Context

- Stage parity report still shows early boundary degradation across chunks and very low decode token match, consistent with the above mechanism.
- Cache validation for current model shows dynamic write path is active and passing.

## Practical Root-Cause Statement

Root cause is ANE linear-attention numerical perturbation amplification (recurrence -> RMSNormGated/out_proj), not dynamic KV cache indexing.
