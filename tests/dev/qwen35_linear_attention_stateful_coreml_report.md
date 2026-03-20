# Qwen3.5 Linear-Attention Stateful CoreML Parity

## Command

```bash
python tests/dev/test_qwen35_linear_attention_stateful_coreml_vs_hf.py \
  --model-path /Users/yw68/local_llm/models/Qwen__Qwen3.5-4B \
  --layer-idx 30 \
  --seq-len 32 \
  --prefill-chunk-len 256 \
  --prompt-lens 128,320,1026
```

## Contract

- Stateful linear-attention uses two CoreML states:
  - `conv_state`
  - `recurrent_state`
- Export contract uses:
  - prefill input length `32` for the base CoreML stateful block
  - decode input length `1`
- Long-prompt validation uses chunked PyTorch prefill with chunk size `256`.

## Single-Chunk PyTorch Parity vs HF

- `prefill_torch_vs_hf`: `max_abs=1.953125e-03`, `mean_abs=4.374756e-05`, `rmse=7.294421e-05`, `cosine=0.99999487`
- `decode_torch_vs_hf`: `max_abs=4.882812e-04`, `mean_abs=5.319435e-05`, `rmse=8.867914e-05`, `cosine=0.99999976`

## Chunked PyTorch Parity vs HF

- `prompt_len=128`
  - `prefill_chunked`: `max_abs=1.953125e-03`, `mean_abs=4.904055e-05`, `rmse=8.142352e-05`, `cosine=1.00000036`
  - `decode_chunked`: `max_abs=4.882812e-04`, `mean_abs=4.607218e-05`, `rmse=7.326296e-05`, `cosine=0.99999934`
- `prompt_len=320`
  - `prefill_chunked`: `max_abs=1.301392e+00`, `mean_abs=2.075269e-02`, `rmse=6.107223e-02`, `cosine=0.94879508`
  - `decode_chunked`: `max_abs=2.953491e-01`, `mean_abs=6.537496e-02`, `rmse=8.286688e-02`, `cosine=0.90661502`
- `prompt_len=1026`
  - `prefill_chunked`: `max_abs=1.301758e+00`, `mean_abs=4.873562e-02`, `rmse=7.876511e-02`, `cosine=0.92192739`
  - `decode_chunked`: `max_abs=7.705078e-01`, `mean_abs=1.539298e-01`, `rmse=1.954428e-01`, `cosine=0.42513317`

## Chunked PyTorch Self-Consistency

- `prompt_len=128`
  - `prefill_self`: `max_abs=0.0`, `mean_abs=0.0`, `rmse=0.0`, `cosine=0.99999249`
  - `decode_self`: `max_abs=0.0`, `mean_abs=0.0`, `rmse=0.0`, `cosine=0.99999976`
- `prompt_len=320`
  - `prefill_self`: `max_abs=0.0`, `mean_abs=0.0`, `rmse=0.0`, `cosine=1.00001085`
  - `decode_self`: `max_abs=0.0`, `mean_abs=0.0`, `rmse=0.0`, `cosine=0.99999952`
- `prompt_len=1026`
  - `prefill_self`: `max_abs=0.0`, `mean_abs=0.0`, `rmse=0.0`, `cosine=1.00027168`
  - `decode_self`: `max_abs=0.0`, `mean_abs=0.0`, `rmse=0.0`, `cosine=0.99999958`

## Conclusions

- The ANEMLL linear-attention implementation is now internally consistent for chunked long-prompt prefill and decode.
- HF remains a good teacher for the single-chunk path.
- HF is not a stable teacher for true chunked long-prompt linear prefill, because its `Qwen3_5GatedDeltaNet` path does not provide a teacher-faithful multi-token state-carry contract across chunks.

## CoreML Status

- Stateful CoreML export succeeds for both prefill and decode blocks.
- CoreML runtime diagnosis on macOS:
  - `CPU_ONLY`: prefill and decode succeed
  - `CPU_AND_GPU`: prefill and decode succeed
  - `ALL`: fails with ANE program inference error during prefill
- ANE isolation probes:
  - non-stateful `projconv`: succeeds on `ALL`
  - non-stateful `recurrent_core`: succeeds on `ALL`
  - non-stateful `normout`: succeeds on `ALL`
  - stateful `conv_state` probe: fails on `ALL`
- Current evidence points to the stateful convolution-state read/update path as the first ANE-specific blocker.
- Stateless CoreML path with explicit tensor I/O for `conv_state` and `recurrent_state` succeeds on macOS:
  - `stateless_linear_prefill`
    - `CPU_ONLY`: OK
    - `CPU_AND_GPU`: OK
    - `ALL`: OK
  - `stateless_linear_decode`
    - `CPU_ONLY`: OK
    - `CPU_AND_GPU`: OK
    - `ALL`: OK
- Stateless CoreML parity vs PyTorch on macOS (`seq_len=32`):
  - `prefill_coreml_vs_torch`: `max_abs=3.537483e-01`, `mean_abs=1.311458e-02`, `rmse=1.753251e-02`, `cosine=0.98612529`
  - `decode_coreml_vs_torch`: `max_abs=7.128906e-02`, `mean_abs=1.432181e-02`, `rmse=1.823837e-02`, `cosine=0.99190891`
- So current status is:
  - PyTorch parity: good
  - chunked self-consistency: good
  - CoreML export: good
  - CoreML CPU/GPU runtime: good
  - Stateful ANE runtime execution: blocked
  - Stateless ANE runtime execution: works
