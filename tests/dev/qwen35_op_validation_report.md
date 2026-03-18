# Qwen3.5 Op-Level Validation Report

Model path: `/home/yue/local_llm/models/Qwen__Qwen3.5-4B`
Environment: `conda activate qwen_coreml`
Date: 2026-03-18

## Full Attention Parity (HF)
Source: `tests/dev/test_qwen35_full_attention_vs_hf.py --layer-idx 3 --seq-len 16`
- max_abs: `1.953125e-03`
- mean_abs: `1.186585e-04`
- rmse: `1.756284e-04`
- cosine: `0.99999726`

## Full Attention Cache Parity (HF)
Source: `tests/dev/test_qwen35_full_attention_cache_vs_hf.py --layer-idx 3 --seq-len 12`
- prefill max_abs: `1.953125e-03`
- prefill mean_abs: `1.074076e-04`
- prefill rmse: `1.642137e-04`
- prefill cosine: `0.99999779`
- decode max_abs: `6.103516e-04`
- decode mean_abs: `1.351487e-04`
- decode rmse: `1.822650e-04`
- decode cosine: `0.99999952`

## Linear Attention Projection Parity (HF)
Source: `tests/dev/test_qwen35_linear_attention_proj_vs_hf.py --layer-idx 0 --seq-len 16`
- in_proj_qkv max_abs: `0.0`
- in_proj_a max_abs: `0.0`
- in_proj_b max_abs: `0.0`
- in_proj_z max_abs: `0.0`
- out_proj max_abs: `0.0`
- conv (functional prefill path) max_abs: `0.0`

## Linear Attention Forward + Cache Parity (HF)
Source: `tests/dev/test_qwen35_linear_attention_vs_hf.py --layer-idx 30 --seq-len 32`
- forward max_abs: `9.765625e-04`
- forward mean_abs: `7.904886e-05`
- forward rmse: `1.112104e-04`
- forward cosine: `0.99999630`
- prefill max_abs: `9.765625e-04`
- prefill mean_abs: `7.904886e-05`
- prefill rmse: `1.112104e-04`
- prefill cosine: `0.99999630`
- decode max_abs: `9.765625e-04`
- decode mean_abs: `9.621556e-05`
- decode rmse: `1.348983e-04`
- decode cosine: `0.99999964`

## Notes
- For isolated layer tests, decode parity should use a last linear-attention layer (layer 30 for this checkpoint) so HF recurrent cache path (`has_previous_state`) is enabled.
- Layer 0 decode in isolation is expected to diverge if recurrent path is not activated in HF cache policy.
