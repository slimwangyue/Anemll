# Qwen3.5-4B Prefill Parity Validation — Agent Runbook

> **Machine**: macOS (Apple Silicon, 16 GB RAM)
> **User**: `yw68`
> **Purpose**: Validate PyTorch ↔ CoreML prefill parity for Qwen3.5-4B (4 chunks).

---

## 1. Environment

```bash
# Python venv (Python 3.12)
source /Users/yw68/Anemll_remote_run/repo/.venv/bin/activate
export PYTHONPATH="/Users/yw68/Anemll_remote_run/repo:$PYTHONPATH"

# Key paths
REPO="/Users/yw68/Anemll_remote_run/repo"
MODEL_PATH="/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
EXPORT_DIR="/Users/yw68/Anemll_remote_run/qwen35_chunk4_export"
TMP_DIR="/tmp/qwen35_prefill_parity"
```

### Installed packages (already available)
- `coremltools`, `torch`, `numpy`, `transformers`

---

## 2. Model Architecture Summary

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 2560 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 20 |
| `num_key_value_heads` | 4 |
| `head_dim` | 128 |
| `intermediate_size` | 9728 |
| `vocab_size` | 151936 |
| `state_length` | 256 (KV cache / context length for this export) |

**Layer types**: Hybrid — most layers are `full_attention`, some are `linear_attention`.
Read `layer_types` from `config.json` to identify which.

### Linear Attention Parameters (from config.json)
- `linear_num_key_heads`, `linear_num_value_heads`
- `linear_key_head_dim`, `linear_value_head_dim`
- `linear_conv_kernel_dim`

Use `Qwen35Config.from_json(MODEL_PATH + "/config.json")` to load all values.

---

## 3. Exported Models in EXPORT_DIR

| File | Part | Notes |
|------|------|-------|
| `qwen35_embeddings.mlpackage` / `.mlmodelc` | Part 1 | Input: `input_ids [1, seq_len]` int32 → Output: `hidden_states [1, seq_len, 2560]` fp16 |
| `qwen35_prefill_chunk_01of04.mlpackage` | Part 2 prefill | Layers 0–7 (8 layers) |
| `qwen35_prefill_chunk_02of04.mlpackage` | Part 2 prefill | Layers 8–15 (8 layers) |
| `qwen35_prefill_chunk_03of04.mlpackage` | Part 2 prefill | Layers 16–23 (8 layers) |
| `qwen35_prefill_chunk_04of04.mlpackage` | Part 2 prefill | Layers 24–31 (8 layers), **last chunk** → output is `[:, 0:1, :]` (1 token) |
| `qwen35_lm_head_lut6.mlpackage` | Part 3 | Input: `hidden_states [1, 1, 2560]` fp16 → Output: logits |

### Prefill Chunk I/O Contract

**Regular inputs** (multiArrayType):
| Name | Shape | dtype |
|------|-------|-------|
| `hidden_states` | `(1, 256, 2560)` | fp16 |
| `position_ids` | `(256,)` | int32 — **rank 1, not rank 2** |
| `causal_mask` | `(1, 1, 256, 256)` | fp16 |
| `current_pos` | `(1,)` | int32 |

**Stateful inputs** (StateType — handled via `model.make_state()`):
| Name | Shape | dtype | Notes |
|------|-------|-------|-------|
| `k_cache` | `(local_layers, 4, 256, 128)` | fp16 | Split K cache per chunk (8 layers) |
| `v_cache` | `(local_layers, 4, 256, 128)` | fp16 | Split V cache per chunk (8 layers) |
| `linear_conv_state` | `(local_layers, conv_dim, conv_kernel)` | fp16 | Only if chunk has linear-attn layers |
| `linear_recurrent_state` | `(local_layers, num_v_heads, key_head_dim, value_head_dim)` | fp16 | Only if chunk has linear-attn layers |

**Output**: `output_hidden_states` — shape `(1, 256, 2560)` for non-final chunks, `(1, 1, 2560)` for last chunk.

> **CRITICAL**: States are **not regular inputs**. You must call `state = model.make_state()` and pass it to `model.predict(inputs, state=state)`. Do NOT pass state tensors in the input dict.

---

## 4. What Has Already Been Done

### PyTorch reference outputs (saved to `/tmp/qwen35_prefill_parity/`)
These files already exist on the Mac:

| File | Content | Shape |
|------|---------|-------|
| `input_ids.npy` | Tokenized prompt (256 tokens) | `(1, 256)` int32 |
| `embed_out.npy` | PyTorch embedding output | `(1, 256, 2560)` fp16 |
| `torch_chunk1.npy` | PyTorch output after layers 0–7 | `(1, 256, 2560)` fp16 |
| `torch_chunk2.npy` | PyTorch output after layers 8–15 | `(1, 256, 2560)` fp16 |
| `torch_chunk3.npy` | PyTorch output after layers 16–23 | `(1, 256, 2560)` fp16 |
| `torch_chunk4.npy` | PyTorch output after layers 24–31 | `(1, 256, 2560)` fp16 |
| `torch_top.npy` | Top-1 token ID from PyTorch | scalar |

**Verified**: Embeddings parity is perfect (`max_abs=0.0, cosine=1.0`).

### What remains
Run each CoreML prefill chunk and compare its output against the corresponding `torch_chunkN.npy`.

---

## 5. Known Issues & Gotchas

### 5.1 Memory (16 GB Mac)
- **DO NOT** load the PyTorch model (~8 GB) and CoreML models in the same process.
- The two-phase approach (save PyTorch outputs first, then compare CoreML separately) avoids OOM segfaults.
- Run **one chunk at a time** — load model, predict, save result, delete model, `gc.collect()`.

### 5.2 ANE Failures
- ANE may reject some models with `ANEProgramProcessRequestDirect() Failed`.
- If ANE fails, retry with `ct.ComputeUnit.CPU_AND_GPU` for comparison numbers — but note this may mask ANE-specific bugs.
- The goal per CLAUDE.md is to always validate on ANE (`CPU_AND_NE`), but CPU_AND_GPU is acceptable as a diagnostic fallback.

### 5.3 `.mlmodelc` vs `.mlpackage`
- `.mlmodelc` (compiled) can only be loaded with `ct.models.CompiledMLModel(path, compute_unit)`.
- `.mlpackage` is loaded with `ct.models.MLModel(path, compute_units=compute_unit)`.
- Note the different API: `compute_unit` (singular, positional) vs `compute_units=` (keyword).
- Prefer `.mlmodelc` for faster load and ANE execution.

### 5.4 Output Shape Mismatch (Chunk 4)
- PyTorch's `process_layers_prefill_export_local_state` with `apply_final_norm=False` returns `(1, 256, 2560)` for ALL chunks.
- But the CoreML chunk 4 model applies `out[:, 0:1, :]` inside the wrapper, returning `(1, 1, 2560)`.
- **The PyTorch reference `torch_chunk4.npy` is the full `(1, 256, 2560)` — so for chunk 4, compare only the first token**: `torch_chunk4[:, 0:1, :]` vs CoreML output `(1, 1, 2560)`.

### 5.5 Cascaded vs Isolated Comparison
There are two comparison modes:
1. **Isolated** (recommended first): Feed `torch_chunkN.npy` (PyTorch output of previous chunk) as input to CoreML chunk N+1. This isolates per-chunk divergence.
2. **Cascaded**: Feed CoreML output of chunk N as input to CoreML chunk N+1. This shows accumulated error but gives realistic end-to-end numbers.

### 5.6 Python stdout buffering
When running via `nohup`, Python buffers stdout. Use `python -u` (unbuffered) or `PYTHONUNBUFFERED=1`.

---

## 6. The Task: Run Prefill Parity

### Step-by-step

```bash
cd /Users/yw68/Anemll_remote_run/repo
source .venv/bin/activate
export PYTHONUNBUFFERED=1
```

### Option A: Use the existing test script

The existing test script is the cleanest approach if OOM doesn't hit:

```bash
python tests/dev/test_qwen35_prefill_chunk_compare.py \
    --model-path /Users/yw68/local_llm/models/Qwen__Qwen3.5-4B \
    --export-dir /Users/yw68/Anemll_remote_run/qwen35_chunk4_export \
    --prefix qwen35 \
    --num-chunks 4 \
    --context-length 256
```

**Warning**: This loads BOTH PyTorch model and CoreML models. On 16 GB Mac, this segfaults. Skip to Option B.

### Option B: Two-phase approach (RECOMMENDED)

#### Phase 1: PyTorch references (ALREADY DONE)
The files in `/tmp/qwen35_prefill_parity/` already contain the PyTorch outputs. Verify they exist:

```bash
ls -la /tmp/qwen35_prefill_parity/
# Should show: input_ids.npy, embed_out.npy, torch_chunk{1,2,3,4}.npy, torch_top.npy
```

If they're missing, regenerate with:

```python
#!/usr/bin/env python3
"""Phase 1: Generate PyTorch reference outputs."""
import torch, numpy as np, os, sys
from pathlib import Path
from transformers import AutoTokenizer
sys.path.insert(0, "/Users/yw68/Anemll_remote_run/repo")
from anemll.models.qwen3_5_model import MODEL_DTYPE, Qwen35Config, Qwen35ForCausalLM

model_path = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
seq_len = 256
num_chunks = 4
tmp_dir = "/tmp/qwen35_prefill_parity"
os.makedirs(tmp_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
prompt = "Explain stack and heap memory in one paragraph and give one debugging tip."
text = prompt
while True:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    if ids.shape[1] >= seq_len:
        ids = ids[:, :seq_len]
        break
    text = text + " " + prompt
np.save(f"{tmp_dir}/input_ids.npy", ids.numpy())

cfg = Qwen35Config.from_json(str(Path(model_path) / "config.json"))
model = Qwen35ForCausalLM(cfg).half().eval()
model.load_pretrained_weights(model_path)

with torch.no_grad():
    torch_hidden = model.model.embed_tokens(ids.to(torch.int32)).to(torch.float16)
np.save(f"{tmp_dir}/embed_out.npy", torch_hidden.numpy())

position_ids = torch.arange(seq_len, dtype=torch.int32)
causal_mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=torch.float16)
row = torch.arange(seq_len).reshape(seq_len, 1)
col = torch.arange(seq_len).reshape(1, seq_len)
causal_mask[:, :, col <= row] = 0
current_pos = torch.tensor(0, dtype=torch.int32)

total_layers = cfg.num_hidden_layers
layers_per = total_layers // num_chunks
chunk_ranges = []
start = 0
for i in range(num_chunks):
    end = start + layers_per if i < num_chunks - 1 else total_layers
    chunk_ranges.append((start, end))
    start = end

hidden = torch_hidden.clone()
with torch.no_grad():
    for ci, (s, e) in enumerate(chunk_ranges):
        local_layers = e - s
        kv = torch.zeros((2 * local_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim), dtype=MODEL_DTYPE)
        conv_dim = (
            cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
            + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
        )
        conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
        conv = torch.zeros((local_layers, conv_dim, conv_kernel), dtype=MODEL_DTYPE)
        rec = torch.zeros((local_layers, cfg.text_config.linear_num_value_heads,
                           cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim), dtype=MODEL_DTYPE)
        hidden = model.model.process_layers_prefill_export_local_state(
            hidden_states=hidden, position_ids=position_ids, causal_mask=causal_mask,
            current_pos=current_pos, kv_cache_0=kv, linear_conv_state=conv,
            linear_recurrent_state=rec, start_layer=s, end_layer=e,
            apply_final_norm=False, expected_batch_size=1, expected_seq_len=seq_len,
        )
        np.save(f"{tmp_dir}/torch_chunk{ci+1}.npy", hidden.numpy())
        print(f"chunk {ci+1} ({s}:{e}) shape={hidden.shape}")

print("Phase 1 done")
```

#### Phase 2: CoreML comparison (THE MAIN TASK)

Run each CoreML chunk one at a time, comparing against saved PyTorch outputs:

```python
#!/usr/bin/env python3
"""Phase 2: Compare CoreML prefill chunks against saved PyTorch outputs.

Usage:
    python prefill_parity_phase2.py                  # Run all 4 chunks
    python prefill_parity_phase2.py --chunk 1        # Run just chunk 1
    python prefill_parity_phase2.py --cascaded       # Feed CoreML output forward (realistic)
    python prefill_parity_phase2.py --cpu-gpu        # Use CPU+GPU instead of ANE
"""
import numpy as np
import gc
import os
import sys
import argparse
import coremltools as ct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=0, help="Run only this chunk (1-4), 0=all")
    parser.add_argument("--cascaded", action="store_true", help="Feed CoreML output forward (realistic)")
    parser.add_argument("--cpu-gpu", action="store_true", help="Use CPU_AND_GPU instead of ANE")
    parser.add_argument("--compiled", action="store_true", help="Use .mlmodelc instead of .mlpackage")
    args = parser.parse_args()

    tmp_dir = "/tmp/qwen35_prefill_parity"
    export_dir = "/Users/yw68/Anemll_remote_run/qwen35_chunk4_export"
    seq_len = 256
    num_chunks = 4

    compute = ct.ComputeUnit.CPU_AND_GPU if args.cpu_gpu else ct.ComputeUnit.CPU_AND_NE
    ext = ".mlmodelc" if args.compiled else ".mlpackage"
    print(f"Compute: {'CPU_AND_GPU' if args.cpu_gpu else 'CPU_AND_NE'}, ext: {ext}")

    # Load PyTorch references
    input_ids = np.load(f"{tmp_dir}/input_ids.npy")
    torch_embed = np.load(f"{tmp_dir}/embed_out.npy")
    torch_chunks = {i+1: np.load(f"{tmp_dir}/torch_chunk{i+1}.npy") for i in range(num_chunks)}
    print("Loaded PyTorch references")

    # Embeddings
    embed_name = f"qwen35_embeddings{ext}"
    embed_path = os.path.join(export_dir, embed_name)
    print(f"\n--- Embeddings: {embed_name} ---")
    if args.compiled:
        embed_model = ct.models.CompiledMLModel(embed_path, compute)
    else:
        embed_model = ct.models.MLModel(embed_path, compute_units=compute)
    embed_out = embed_model.predict({"input_ids": input_ids.astype(np.int32)})
    cml_embed = list(embed_out.values())[0]

    diff = np.abs(torch_embed.astype(np.float32) - cml_embed.astype(np.float32))
    cos = _cosine(torch_embed, cml_embed)
    print(f"EMBED: max_abs={diff.max():.6f}  mean_abs={diff.mean():.6f}  cosine={cos:.10f}")
    del embed_model; gc.collect()

    # Determine which chunks to run
    chunks_to_run = range(1, num_chunks + 1) if args.chunk == 0 else [args.chunk]

    # For cascaded mode or multi-chunk, we need previous output
    prev_hidden = cml_embed.copy()  # Start from CoreML embed

    for ci in chunks_to_run:
        chunk_name = f"qwen35_prefill_chunk_{ci:02d}of{num_chunks:02d}{ext}"
        chunk_path = os.path.join(export_dir, chunk_name)
        print(f"\n--- Chunk {ci}: {chunk_name} ---")

        # Determine input hidden states
        if args.cascaded or ci == 1:
            # Cascaded: use previous CoreML output
            # Chunk 1: always use embed output
            if ci == 1:
                hidden_input = cml_embed.copy()
            else:
                hidden_input = prev_hidden.copy()
            input_source = "CoreML (cascaded)"
        else:
            # Isolated: use PyTorch reference from previous chunk
            if ci == 1:
                hidden_input = torch_embed.copy()
            else:
                hidden_input = torch_chunks[ci - 1].copy()
            input_source = "PyTorch (isolated)"
        print(f"  Input source: {input_source}, shape={hidden_input.shape}")

        # Load model
        if args.compiled:
            model = ct.models.CompiledMLModel(chunk_path, compute)
        else:
            model = ct.models.MLModel(chunk_path, compute_units=compute)

        # Get spec to build correct inputs
        spec = model.get_spec()
        inp_dict = {}
        for inp in spec.description.input:
            name = inp.name
            if not inp.type.HasField("multiArrayType"):
                continue  # Skip StateType inputs
            shape = tuple(inp.type.multiArrayType.shape)
            if name == "hidden_states":
                inp_dict[name] = hidden_input.astype(np.float16).reshape(shape)
            elif "position" in name:
                inp_dict[name] = np.arange(seq_len, dtype=np.int32).reshape(shape)
            elif "causal_mask" in name or "mask" in name:
                mask = np.full(shape, -65504.0, dtype=np.float16)
                for r in range(shape[-2]):
                    mask[..., r, :r+1] = 0
                inp_dict[name] = mask
            elif "current_pos" in name:
                inp_dict[name] = np.zeros(shape, dtype=np.int32)
            else:
                print(f"  WARNING: unknown input '{name}' shape={shape}, using zeros")
                inp_dict[name] = np.zeros(shape, dtype=np.float16)

        # Create state for stateful inputs (kv_cache, conv_state, etc.)
        state = model.make_state()

        # Run prediction
        try:
            out = model.predict(inp_dict, state=state)
        except RuntimeError as e:
            print(f"  PREDICT FAILED: {e}")
            if not args.cpu_gpu:
                print("  Retrying with CPU_AND_GPU...")
                del model, state; gc.collect()
                if args.compiled:
                    model = ct.models.CompiledMLModel(chunk_path, ct.ComputeUnit.CPU_AND_GPU)
                else:
                    model = ct.models.MLModel(chunk_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
                state = model.make_state()
                out = model.predict(inp_dict, state=state)
            else:
                raise

        cml_hidden = out[list(out.keys())[0]]
        print(f"  Output shape: {cml_hidden.shape}")

        # Save for next chunk (cascaded mode)
        prev_hidden = cml_hidden.copy()

        # Compare against PyTorch reference
        th = torch_chunks[ci]
        if cml_hidden.shape != th.shape:
            print(f"  Shape mismatch: torch={th.shape} cml={cml_hidden.shape}")
            # Last chunk returns (1, 1, 2560) but torch has (1, 256, 2560)
            if cml_hidden.shape[1] < th.shape[1]:
                th_cmp = th[:, :cml_hidden.shape[1], :]  # Compare first token only
                cml_cmp = cml_hidden
                print(f"  Comparing first {cml_hidden.shape[1]} token(s) only")
            else:
                min_seq = min(th.shape[1], cml_hidden.shape[1])
                th_cmp = th[:, :min_seq, :]
                cml_cmp = cml_hidden[:, :min_seq, :]
        else:
            th_cmp = th
            cml_cmp = cml_hidden

        diff = np.abs(th_cmp.astype(np.float32) - cml_cmp.astype(np.float32))
        cos = _cosine(th_cmp, cml_cmp)
        flat = diff.flatten()

        print(f"CHUNK {ci}: max_abs={diff.max():.4f}  mean_abs={diff.mean():.6f}  cosine={cos:.10f}")
        print(f"  p99={np.percentile(flat, 99):.4f}  p95={np.percentile(flat, 95):.4f}  p50={np.percentile(flat, 50):.6f}")

        # Top-5 worst positions
        worst = np.argsort(flat)[-5:][::-1]
        for w in worst:
            pos = np.unravel_index(w, diff.shape)
            print(f"  worst: pos={pos} torch={th_cmp[pos]:.4f} cml={cml_cmp[pos]:.4f} diff={diff[pos]:.4f}")

        del model, state; gc.collect()

    print("\n=== PARITY VALIDATION COMPLETE ===")


def _cosine(a, b):
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_f, b_f) / denom)


if __name__ == "__main__":
    main()
```

### Running the script

```bash
# Full run, ANE, one chunk at a time (safest on 16 GB):
python -u prefill_parity_phase2.py --chunk 1
python -u prefill_parity_phase2.py --chunk 2
python -u prefill_parity_phase2.py --chunk 3
python -u prefill_parity_phase2.py --chunk 4

# Or all at once (loads/unloads one at a time):
python -u prefill_parity_phase2.py

# With CPU+GPU fallback:
python -u prefill_parity_phase2.py --cpu-gpu

# Cascaded (realistic end-to-end):
python -u prefill_parity_phase2.py --cascaded

# Using compiled models (faster):
python -u prefill_parity_phase2.py --compiled
```

---

## 7. Interpreting Results

### Pass criteria
| Metric | Good | Acceptable | Bad |
|--------|------|------------|-----|
| `max_abs` | < 0.5 | < 2.0 | > 5.0 |
| `mean_abs` | < 0.01 | < 0.05 | > 0.1 |
| `cosine` | > 0.999 | > 0.99 | < 0.95 |

### Expected behaviors
- **Chunk 1–3**: These are non-final chunks, output shape `(1, 256, 2560)`. If they're bad, the issue is in that specific chunk's layers.
- **Chunk 4**: Last chunk applies `[:, 0:1, :]` in the wrapper. Only 1 token is compared. This chunk contains the final layers (24–31) which may include full-attention layers with KV cache stateful writes.
- **Cascaded worse than isolated**: Normal — errors accumulate. If isolated is good but cascaded is bad, the per-chunk error compounds.

### What to investigate if results are bad
1. **High max_abs in chunk N**: The layers in that chunk have divergence. Check which layers are `full_attention` vs `linear_attention`.
2. **Linear attention chunks**: Known issue — the 5-stage split (Proj→Conv→Layout→Core→Norm) greatly improves ANE parity. If the converter doesn't use staged export, linear chunks will be worse.
3. **Full attention KV cache**: Known issue — CoreML `StateType` has bugs with dynamic-index writes at position > 0. For prefill (position 0 start), this may be OK.
4. **ANE failure**: If ANE rejects the model entirely, try `--cpu-gpu`. If CPU+GPU works but ANE doesn't, it's an ANE lowering issue.

---

## 8. Re-exporting Models (if needed)

If you need to re-export prefill chunks (e.g., after code changes):

```bash
cd /Users/yw68/Anemll_remote_run/repo
source .venv/bin/activate

# Export all 4 prefill chunks
python -m anemll.ane_converter.qwen3_5_converter \
    --model /Users/yw68/local_llm/models/Qwen__Qwen3.5-4B \
    --output /Users/yw68/Anemll_remote_run/qwen35_chunk4_export \
    --part 2_prefill \
    --context 256 \
    --batch 256 \
    --chunk 4

# Compile to .mlmodelc (optional, for faster loading)
for f in /Users/yw68/Anemll_remote_run/qwen35_chunk4_export/qwen35_prefill_chunk_*.mlpackage; do
    xcrun coremlcompiler compile "$f" "$(dirname "$f")"
done
```

### Export Part 1 & Part 3 (if missing)

```bash
# Embeddings
python -m anemll.ane_converter.qwen3_5_converter \
    --model /Users/yw68/local_llm/models/Qwen__Qwen3.5-4B \
    --output /Users/yw68/Anemll_remote_run/qwen35_chunk4_export \
    --part 1 --context 256 --batch 256

# LM Head
python -m anemll.ane_converter.qwen3_5_converter \
    --model /Users/yw68/local_llm/models/Qwen__Qwen3.5-4B \
    --output /Users/yw68/Anemll_remote_run/qwen35_chunk4_export \
    --part 3 --context 256 --batch 256 --lut 6
```

---

## 9. Source Code Map

| File | Location | Purpose |
|------|----------|---------|
| Model implementation | `anemll/models/qwen3_5_model.py` | Qwen3.5 model, attention, linear attention |
| CoreML converter | `anemll/ane_converter/qwen3_5_converter.py` | Export to CoreML (PrefillWrapper inside) |
| Existing parity test | `tests/dev/test_qwen35_prefill_chunk_compare.py` | Full parity test (OOM on 16GB) |
| PyTorch ref generator | See Phase 1 script above | Saves .npy files |
| CoreML comparator | See Phase 2 script above | Loads .npy + CoreML, compares |

### Key methods in `qwen3_5_model.py`
- `process_layers_prefill_export_local_state()` — processes a range of layers with chunk-local state
- `_process_layer_prefill_export_local_state()` — processes a single layer (dispatches full vs linear attn)
- `Qwen35LinearAttention.forward_prefill_export()` — linear attention forward for prefill

### Key classes in `qwen3_5_converter.py`
- `PrefillWrapper` — wraps model for tracing, owns state buffers, handles last-chunk slicing
- `GetChunkLocalTransformerStates()` — creates `ct.StateType` list for CoreML export

---

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| Segfault / killed | Memory. Run one chunk at a time. Never load PyTorch + CoreML together. |
| `ANEProgramProcessRequestDirect() Failed` | ANE rejected the model. Try `--cpu-gpu`. Check if the mlpackage was exported with `CPU_AND_NE`. |
| `feature 'position_ids' must be of rank 1` | position_ids shape is `(256,)` not `(1, 256)`. Read shape from spec. |
| `input feature for kv_cache_0 must be an MLState` | States need `model.make_state()`, not dict entries. |
| `.mlmodelc` RuntimeError manifest | Use `ct.models.CompiledMLModel(path, compute_unit)` not `ct.models.MLModel()`. |
| Empty output with nohup | Use `python -u` or `PYTHONUNBUFFERED=1`. |
| `torch_chunk4.npy` shape mismatch | PyTorch saves `(1, 256, 2560)` but CoreML returns `(1, 1, 2560)`. Compare `torch[:, 0:1, :]` only. |
