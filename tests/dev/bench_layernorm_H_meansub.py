#!/usr/bin/env python3
"""Test: F.layer_norm(H) with mean subtraction — ANE-compatible RMSNorm.

Instead of:
  A) reduce_mean(x²) + rsqrt  (current — fails on iPhone A16 prefill)
  B) doubled-concat F.layer_norm(2H) (old — 40% slower, 2× overhead)

This uses:
  C) mean-subtracted F.layer_norm(H):
       hidden_states -= mean
       return F.layer_norm(hidden_states, (H,), weight, bias=None, eps=eps)

     - H=2560 stays within ANE's native layer_norm limit
     - No 2× concat overhead
     - Mathematically LayerNorm (not pure RMSNorm) but cos ≥ 0.99 expected
     - Should generate MIL `layer_norm` ops (not `reduce_mean`)

Exports chunk 0 (decode + prefill), combines into deduped chunk,
and verifies ANE loading.
"""
import sys, os, time, gc, warnings
warnings.filterwarnings('ignore')

REPO_ROOT = '/Volumes/MySSD/Anemll'
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts_qwen3_5'))
os.chdir(REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from collections import Counter

torch.set_grad_enabled(False)

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, Qwen35RMSNorm, Qwen35RMSNormGated,
    MODEL_DTYPE, TEST_DEVICE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from anemll.utils.combine_models import _save_multifunction_dedup

HF_MODEL = os.path.join(REPO_ROOT, 'models', 'Qwen__Qwen3.5-4B')
OUT_DIR = os.path.join(REPO_ROOT, 'artifacts', 'layernorm_H_meansub')
HIDDEN = 2560
CHUNK_IDX = 0
CHUNK_START, CHUNK_END = CHUNK_RANGES[CHUNK_IDX]

os.makedirs(OUT_DIR, exist_ok=True)

# ── Save originals ──
orig_rmsnorm_fwd = Qwen35RMSNorm.forward
orig_rmsnormgated_fwd = Qwen35RMSNormGated.forward


# ── New implementations: F.layer_norm(H) with mean subtraction ──
def layernorm_H_meansub_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """RMSNorm approximation via mean-subtracted F.layer_norm(H).

    1. Subtract mean to center (makes LayerNorm ≈ RMSNorm)
    2. F.layer_norm on H dim (stays within ANE's native limit)
    3. Apply (1 + weight) scaling (Qwen3.5 offset semantics)

    Operates in float32 to avoid MIL eps/gamma dtype mismatch.
    """
    orig_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    mean = hidden_states.mean(-1, keepdim=True)
    hidden_states = hidden_states - mean
    w = (1.0 + self.weight).float()
    out = F.layer_norm(hidden_states, (self.hidden_size,), w, bias=None, eps=float(self.eps))
    return out.to(orig_dtype)


def layernorm_H_meansub_gated_forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Gated RMSNorm approximation via mean-subtracted F.layer_norm(H)."""
    orig_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    mean = hidden_states.mean(-1, keepdim=True)
    hidden_states = hidden_states - mean
    normed = F.layer_norm(hidden_states, (self.hidden_size,), self.weight.float(), bias=None, eps=float(self.eps))
    normed = normed.to(orig_dtype)
    return normed * F.silu(gate.to(orig_dtype))


def patch_norms():
    Qwen35RMSNorm.forward = layernorm_H_meansub_forward
    Qwen35RMSNormGated.forward = layernorm_H_meansub_gated_forward


def restore_norms():
    Qwen35RMSNorm.forward = orig_rmsnorm_fwd
    Qwen35RMSNormGated.forward = orig_rmsnormgated_fwd


# ── Load model ──
print("Loading model weights...")
t0 = time.time()
cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
cfg.context_length = CTX
cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(HF_MODEL), "Failed to load weights"
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"  Loaded in {time.time()-t0:.1f}s")

# ── Step 1: Correctness check (PyTorch level) ──
print(f"\n=== CORRECTNESS CHECK (PyTorch) ===")
layer = model.model.layers[0]
test_input = torch.randn(1, 1, HIDDEN, dtype=MODEL_DTYPE, device=TEST_DEVICE) * 0.1

# Test Qwen35RMSNorm
norm = layer.input_layernorm  # Qwen35RMSNorm
out_orig = orig_rmsnorm_fwd(norm, test_input)
out_new = layernorm_H_meansub_forward(norm, test_input)
cos_norm = float(F.cosine_similarity(out_orig.flatten().float(), out_new.flatten().float(), dim=0))
print(f"  Qwen35RMSNorm: cos = {cos_norm:.8f}")

# Test Qwen35RMSNormGated
gnorm = layer.self_attn.core_norm_stage.norm  # Qwen35RMSNormGated
test_gate = torch.randn_like(test_input[:, :, :gnorm.hidden_size])
test_h = torch.randn(1, 1, gnorm.hidden_size, dtype=MODEL_DTYPE, device=TEST_DEVICE) * 0.1
out_g_orig = orig_rmsnormgated_fwd(gnorm, test_h, test_gate)
out_g_new = layernorm_H_meansub_gated_forward(gnorm, test_h, test_gate)
cos_gated = float(F.cosine_similarity(out_g_orig.flatten().float(), out_g_new.flatten().float(), dim=0))
print(f"  Qwen35RMSNormGated: cos = {cos_gated:.8f}")

# ── Step 2: Export chunk decode + prefill ──
dec_path = os.path.join(OUT_DIR, f'ffn_LUT4_chunk{CHUNK_IDX}.mlpackage')
pf_path = os.path.join(OUT_DIR, f'prefill_LUT4_chunk{CHUNK_IDX}.mlpackage')

skip_export = '--skip-export' in sys.argv or '--skip-existing' in sys.argv

if not skip_export or not os.path.exists(dec_path):
    print(f"\n=== EXPORT DECODE chunk {CHUNK_IDX} (layers {CHUNK_START}-{CHUNK_END-1}) ===")
    patch_norms()
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
        compute_precision="float16",
    )
    t0 = time.time()
    ml = conv.convert_part_2(
        model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS,
        override_start_layer=CHUNK_START, override_end_layer=CHUNK_END,
    )
    ml.save(dec_path)
    print(f"  Saved decode in {time.time()-t0:.1f}s")
    del ml, conv; gc.collect()
    restore_norms()
else:
    print(f"\n  Decode: cached at {dec_path}")

if not skip_export or not os.path.exists(pf_path):
    print(f"\n=== EXPORT PREFILL chunk {CHUNK_IDX} (layers {CHUNK_START}-{CHUNK_END-1}) ===")
    patch_norms()
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
        compute_precision="float16",
    )
    t0 = time.time()
    ml = conv.convert_part_2_prefill(
        model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS,
        override_start_layer=CHUNK_START, override_end_layer=CHUNK_END,
    )
    ml.save(pf_path)
    print(f"  Saved prefill in {time.time()-t0:.1f}s")
    del ml, conv; gc.collect()
    restore_norms()
else:
    print(f"\n  Prefill: cached at {pf_path}")

del model; gc.collect()

# ── Step 3: MIL op analysis ──
print(f"\n=== MIL OP ANALYSIS ===")
for label, path in [('decode', dec_path), ('prefill', pf_path)]:
    spec = ct.utils.load_spec(path)
    prog = spec.mlProgram
    for fn in prog.functions:
        func = prog.functions[fn]
        for k in func.block_specializations:
            block = func.block_specializations[k]
            break
        break
    counts = Counter()
    for op in block.operations:
        counts[op.type] += 1
    total = sum(counts.values())
    weight = sum(v for k, v in counts.items() if k in ('const', 'constexpr_lut_to_dense'))
    compute = total - weight
    ln = counts.get('layer_norm', 0)
    rm = counts.get('reduce_mean', 0)
    rs = counts.get('rsqrt', 0)
    cv = counts.get('conv', 0)
    tr = counts.get('transpose', 0)
    print(f"  {label:8s}: compute={compute:4d}  layer_norm={ln:2d}  reduce_mean={rm:2d}  rsqrt={rs:2d}  conv={cv:2d}  trans={tr:2d}")

# ── Step 4: Combine into deduped chunk ──
combined_path = os.path.join(OUT_DIR, f'chunk{CHUNK_IDX}.mlpackage')
if not os.path.exists(combined_path) or '--force-combine' in sys.argv:
    print(f"\n=== COMBINE (dedup) chunk {CHUNK_IDX} ===")
    sources = [
        (dec_path, "main", "infer"),
        (pf_path, "main", "prefill"),
    ]
    t0 = time.time()
    _save_multifunction_dedup(sources, combined_path, dedup_weights=True, verbose=False)
    # Size
    total_bytes = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(combined_path)
        for fn in fns
        if not os.path.islink(os.path.join(dp, fn))
    )
    print(f"  Combined chunk: {total_bytes / (1024**2):.1f} MB ({time.time()-t0:.1f}s)")
else:
    print(f"\n  Combined: cached at {combined_path}")

# ── Step 5: ANE loading test ──
print(f"\n=== ANE LOADING TEST ===")
for fn_name in ['infer', 'prefill']:
    for cu_label, cu in [('CPU_AND_NE', ct.ComputeUnit.CPU_AND_NE), ('CPU_ONLY', ct.ComputeUnit.CPU_ONLY)]:
        label = f'{fn_name} {cu_label}'
        print(f"  {label:30s}... ", end='', flush=True)
        try:
            m = ct.models.MLModel(combined_path, compute_units=cu, function_name=fn_name)
            state = m.make_state()
            print("OK")
            del m, state; gc.collect()
        except Exception as e:
            print(f"FAIL: {str(e)[:100]}")

# ── Step 6: Quick inference test (decode only) ──
print(f"\n=== QUICK INFERENCE TEST (decode) ===")
nl = CHUNK_END - CHUNK_START
np.random.seed(42)
inputs = {
    'hidden_states': np.random.randn(1, 1, HIDDEN).astype(np.float16) * 0.01,
    'position_ids': np.array([0], dtype=np.int32),
    'causal_mask': np.zeros((1, 1, 1, CTX), dtype=np.float16),
    'current_pos': np.array([0], dtype=np.int32),
    'linear_conv_state': np.zeros((nl, 1024, 32), dtype=np.float16),
    'linear_recurrent_state': np.zeros((nl, 32, 128, 128), dtype=np.float16),
}

try:
    m = ct.models.MLModel(combined_path, compute_units=ct.ComputeUnit.CPU_AND_NE, function_name='infer')
    state = m.make_state()
    out = m.predict(inputs, state=state)
    h_out = np.asarray(out['output_hidden_states'])
    print(f"  Output shape: {h_out.shape}, range: [{h_out.min():.4f}, {h_out.max():.4f}]")
    print(f"  Non-zero: {np.count_nonzero(h_out)}/{h_out.size}")

    # Timing (5 warmup + 20 runs)
    import resource
    for _ in range(5):
        m.predict(inputs, state=state)
    times = []
    for _ in range(20):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        m.predict(inputs, state=state)
        wall = (time.perf_counter() - t0) * 1000
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = ((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)) * 1000
        times.append((wall, cpu))
    w = np.median([t[0] for t in times])
    c = np.median([t[1] for t in times])
    a = max(0, w - c)
    pct = a / w * 100 if w > 0 else 0
    print(f"  Timing: wall={w:.2f}ms  cpu={c:.2f}ms  ane={a:.2f}ms ({pct:.0f}%)")
    del m; gc.collect()
except Exception as e:
    print(f"  FAILED: {e}")

print(f"\nDone. Artifacts at: {OUT_DIR}")
