# Qwen3.5 Linear Model-Level Validation (ANEMLL vs HF)

- model_path: `/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B`
- fixed_state_length: `256`
- max_new_tokens: `8`
- linear_mode: `split4_ref`
- input policy: both HF and ANEMLL use the same fixed-window-truncated prompt

| mode | prompt | full_tokens | used_tokens | prefill_cos | decode_cos | top1 | text_sim |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal | short | 25 | 25 | 1.0005 | 0.9998 | 1.0000 | 1.0000 |
| normal | medium | 32 | 32 | 1.0010 | 0.9999 | 1.0000 | 1.0000 |
| think | short | 23 | 23 | 1.0005 | 0.9997 | 1.0000 | 1.0000 |
| think | medium | 30 | 30 | 1.0009 | 0.9998 | 1.0000 | 1.0000 |

## normal / short

- full_tokens: `25`
- used_tokens: `25`
- prefill: `prefill              max_abs=3.476562e-01 mean_abs=2.245302e-02 rmse=3.100456e-02 cosine=1.00054622 kl=3.124474e-04 top1=1.000000e+00`
- decode_mean: `decode_mean          max_abs=1.416016e-01 mean_abs=2.340086e-02 rmse=2.919096e-02 cosine=0.99984485 kl=5.448700e-05 top1=1.000000e+00`
- text_similarity: `1.0000`
- HF answer:
```text
4
```
- ANEMLL answer:
```text
4
```

## normal / medium

- full_tokens: `32`
- used_tokens: `32`
- prefill: `prefill              max_abs=3.066406e-01 mean_abs=2.027172e-02 rmse=2.747769e-02 cosine=1.00101304 kl=3.719969e-04 top1=1.000000e+00`
- decode_mean: `decode_mean          max_abs=1.323242e-01 mean_abs=1.865004e-02 rmse=2.346668e-02 cosine=0.99993924 kl=4.620795e-04 top1=1.000000e+00`
- text_similarity: `1.0000`
- HF answer:
```text
### Stack vs. Heap: The Simple
```
- ANEMLL answer:
```text
### Stack vs. Heap: The Simple
```

## think / short

- full_tokens: `23`
- used_tokens: `23`
- prefill: `prefill              max_abs=3.525391e-01 mean_abs=2.156061e-02 rmse=2.955830e-02 cosine=1.00045443 kl=3.530969e-04 top1=1.000000e+00`
- decode_mean: `decode_mean          max_abs=4.001617e-01 mean_abs=5.779009e-02 rmse=7.343390e-02 cosine=0.99971016 kl=5.385674e-05 top1=1.000000e+00`
- text_similarity: `1.0000`
- HF answer:
```text
Thinking Process:

1.  **
```
- ANEMLL answer:
```text
Thinking Process:

1.  **
```

## think / medium

- full_tokens: `30`
- used_tokens: `30`
- prefill: `prefill              max_abs=3.066406e-01 mean_abs=1.997906e-02 rmse=2.699593e-02 cosine=1.00089824 kl=4.030785e-04 top1=1.000000e+00`
- decode_mean: `decode_mean          max_abs=2.849016e-01 mean_abs=4.475864e-02 rmse=5.603600e-02 cosine=0.99977863 kl=4.617690e-05 top1=1.000000e+00`
- text_similarity: `1.0000`
- HF answer:
```text
Thinking Process:

1.  **
```
- ANEMLL answer:
```text
Thinking Process:

1.  **
```
