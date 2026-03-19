# Qwen3.5 Full-Attention Stateful CoreML Parity (Chunked Prompt Validation)

## Command

```bash
python tests/dev/test_qwen35_full_attention_stateful_coreml_vs_hf.py \
  --model-path /Users/yw68/local_llm/models/Qwen__Qwen3.5-4B \
  --layer-idx 3 \
  --input-seq-len 256 \
  --cache-size 2048 \
  --prompt-lengths 128,320,1026
```

## Contract

- Fixed prefill input size: `256`
- Fixed stateful KV cache size: `2048`
- Prompt prefill is split into `256`-token chunks.
- Each chunk advances `current_pos` by the real prompt tokens in that chunk, not by the padded chunk width.
- The final chunk is zero-padded to `256` for the fixed-shape CoreML contract.

## Chunked PyTorch Parity vs HF

- `prompt_len=128`
  - `prefill_torch_vs_hf`: `max_abs=3.906250e-03`, `mean_abs=1.259862e-04`, `rmse=1.706889e-04`, `cosine=1.00000668`
  - `decode_torch_vs_hf`: `max_abs=9.765625e-04`, `mean_abs=1.206633e-04`, `rmse=1.572378e-04`, `cosine=0.99999887`
- `prompt_len=320`
  - `prefill_torch_vs_hf`: `max_abs=1.953125e-03`, `mean_abs=1.334768e-04`, `rmse=1.778331e-04`, `cosine=1.00004458`
  - `decode_torch_vs_hf`: `max_abs=1.098633e-03`, `mean_abs=2.697159e-04`, `rmse=3.392269e-04`, `cosine=0.99999535`
- `prompt_len=1026`
  - `prefill_torch_vs_hf`: `max_abs=6.347656e-03`, `mean_abs=1.976349e-04`, `rmse=2.702426e-04`, `cosine=1.00042057`
  - `decode_torch_vs_hf`: `max_abs=1.403809e-03`, `mean_abs=2.652436e-04`, `rmse=3.353100e-04`, `cosine=0.99998629`

## Chunked CoreML Runtime Parity vs PyTorch

- `prompt_len=128`
  - `prefill_coreml_vs_torch`: `max_abs=3.906250e-02`, `mean_abs=5.606603e-04`, `rmse=7.306273e-04`, `cosine=0.99999774`
  - `decode_coreml_vs_torch`: `max_abs=3.906250e-03`, `mean_abs=5.118483e-04`, `rmse=6.429493e-04`, `cosine=0.99999177`
- `prompt_len=320`
  - `prefill_coreml_vs_torch`: `max_abs=2.343750e-02`, `mean_abs=4.942394e-04`, `rmse=6.351196e-04`, `cosine=1.00003719`
  - `decode_coreml_vs_torch`: `max_abs=1.953125e-03`, `mean_abs=4.398242e-04`, `rmse=5.561428e-04`, `cosine=0.99999011`
- `prompt_len=1026`
  - `prefill_coreml_vs_torch`: `max_abs=4.687500e-02`, `mean_abs=4.434796e-04`, `rmse=5.667447e-04`, `cosine=1.00041437`
  - `decode_coreml_vs_torch`: `max_abs=1.815796e-03`, `mean_abs=3.963817e-04`, `rmse=4.983477e-04`, `cosine=0.99997103`

## Notes

- The chunked validation uses multiple fixed-shape stateful prefill blocks, one per chunk start position encountered in the test prompt lengths, plus a decode block at the final prompt position.
- The validation confirms that carrying `k_cache` and `v_cache` across chunked prefill calls remains numerically stable for prompts longer than the fixed `256`-token input size.
- The script follows Apple stateful conversion guidance (`ct.StateType`, `states=[...]`, `minimum_deployment_target=iOS18`):
  - https://apple.github.io/coremltools/docs-guides/source/stateful-models.html
