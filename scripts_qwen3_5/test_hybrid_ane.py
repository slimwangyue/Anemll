#!/usr/bin/env python3
"""Test: export chunk0 prefill with force_recurrent=True, convert to CoreML, load on ANE.

This validates that the hybrid approach (batched full_attention + unrolled
sequential linear_attention inside one predict() call) actually compiles and
runs on the Apple Neural Engine.

Approach: monkey-patch _forward_prefill_export_impl to set force_recurrent=True,
then use the exact same PrefillWrapper from the converter. This ensures we test
the production code path with minimal changes.

Skips LUT quantization to save time — we only care about ANE loadability.
"""
import sys, os, time, json, warnings
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anemll.models.qwen3_5_model import (
    Qwen35Config, Qwen35ForCausalLM, Qwen35LinearAttention,
    MODEL_DTYPE, TEST_DEVICE, ane_conv_state_shape,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
import coremltools as ct

# ── Config ───────────────────────────────────────────────────────
MODEL_PATH = '/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B'
EXISTING_CHUNK0 = '/Users/yw68/Anemll/qwen3_5_stable_models_6chunk/combined_LUT6_dedup/chunk0.mlpackage'
SEQ_LEN = 512     # Must match BATCH_SIZE used in existing models
CTX_LEN = 2048
NUM_CHUNKS = 6
CHUNK_IDX = 0

cu = ct.ComputeUnit.CPU_AND_NE

# ── Monkey-patch: force_recurrent=True in prefill export ─────────
_orig_forward_prefill_export_impl = Qwen35LinearAttention._forward_prefill_export_impl

def _patched_forward_prefill_export_impl(
    self,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    has_previous_state: bool = True,
    force_fp16_math: bool = False,
    valid_len: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as original but force_recurrent=True in core_norm_stage call."""
    bsz = self.export_expected_batch_size
    seq_len = self.export_expected_seq_len
    mixed_qkv_pre, z_cf, b_cf, a_cf = self.proj_stage(hidden_states)
    conv_out_cf, next_conv_state = self.conv_stage(
        mixed_qkv_pre, conv_state, expected_seq_len=seq_len,
        valid_len=valid_len,
    )
    query, key, value, g, beta, z = self.layout_stage(
        conv_out_cf, z_cf, b_cf, a_cf, bsz, seq_len,
        force_fp16_math=force_fp16_math,
    )
    if valid_len is not None:
        positions = torch.arange(seq_len, device=key.device, dtype=valid_len.dtype)
        valid_mask = (positions < valid_len).to(key.dtype)
        mask_bsh1 = valid_mask.reshape(1, seq_len, 1, 1)
        mask_bsh = valid_mask.reshape(1, seq_len, 1)
        key = key * mask_bsh1
        value = value * mask_bsh1
        beta = beta * mask_bsh
        g = g * mask_bsh
    out, next_recurrent_state = self.core_norm_stage(
        query=query, key=key, value=value, g=g, beta=beta, z=z,
        recurrent_state=recurrent_state,
        has_previous_state=has_previous_state,
        bsz=bsz, seq_len=seq_len,
        force_recurrent=True,          # ← THE ONLY CHANGE
        force_fp16_math=force_fp16_math,
    )
    return out, next_conv_state, next_recurrent_state

Qwen35LinearAttention._forward_prefill_export_impl = _patched_forward_prefill_export_impl
print('[PATCH] force_recurrent=True applied to _forward_prefill_export_impl')

# ── Load config ──────────────────────────────────────────────────
print('=' * 70)
print('STEP 1: Load model config and compute chunk0 layer range')
print('=' * 70)
cfg_path = os.path.join(MODEL_PATH, 'config.json')
with open(cfg_path) as f:
    raw = json.load(f)
cfg = Qwen35Config(raw)
cfg.state_length = CTX_LEN   # Must match export — default 256 is too small for seq_len=512

total_layers = cfg.num_hidden_layers
base, rem = divmod(total_layers, NUM_CHUNKS)
start_layer = CHUNK_IDX * base + min(CHUNK_IDX, rem)
end_layer = start_layer + base + (1 if CHUNK_IDX < rem else 0)
local_num_layers = end_layer - start_layer
layer_types = cfg.text_config.layer_types[start_layer:end_layer]
is_last_chunk = (end_layer >= total_layers)

print(f'  Total layers: {total_layers}')
print(f'  Chunk0: layers {start_layer}-{end_layer-1} ({local_num_layers} layers)')
print(f'  Types: {layer_types}')
print(f'  Is last chunk: {is_last_chunk}')

# ── Load full model weights ──────────────────────────────────────
print(f'\n{"="*70}')
print('STEP 2: Load Qwen3.5-4B weights')
print('=' * 70)
t0 = time.time()
model = Qwen35ForCausalLM(cfg)
model.load_pretrained_weights(MODEL_PATH)
model.eval()
print(f'  Loaded in {time.time()-t0:.1f}s')

# ── Build PrefillWrapper (reuse exact converter code) ────────────
print(f'\n{"="*70}')
print('STEP 3: Build PrefillWrapper (same as converter, with patched recurrence)')
print('=' * 70)

# Inline the PrefillWrapper from the converter — exact same class
class PrefillWrapper(torch.nn.Module):
    def __init__(self, model, start_layer, end_layer, export_seq_len):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.export_seq_len = export_seq_len
        self.local_num_layers = end_layer - start_layer
        self._is_last_chunk = end_layer == len(model.model.layers)
        self._hidden_size = model.config.hidden_size
        cfg = model.config
        self.register_buffer("k_cache", torch.zeros(
            (self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
            dtype=MODEL_DTYPE, device=TEST_DEVICE))
        self.register_buffer("v_cache", torch.zeros(
            (self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
            dtype=MODEL_DTYPE, device=TEST_DEVICE))
        if cfg.has_linear_attention():
            conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
            conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
            ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
            self._lin_conv_shape = (self.local_num_layers, ane_dim1, ane_dim2)
            self._lin_rec_shape = (
                self.local_num_layers,
                cfg.text_config.linear_num_value_heads,
                cfg.text_config.linear_key_head_dim,
                cfg.text_config.linear_value_head_dim,
            )
            self._has_linear = True
        else:
            self._has_linear = False
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = model.model.layers[layer_idx]
            if getattr(layer, "layer_type", None) == "linear_attention":
                layer.self_attn.export_expected_batch_size = 1
                layer.self_attn.export_expected_seq_len = export_seq_len
        self.states = Qwen35Converter.GetChunkLocalTransformerStates(
            model, self.local_num_layers, prefix="", split_full_attention_kv=True)

    def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                linear_conv_state, linear_recurrent_state, valid_len):
        # Uses process_layers_prefill_export_local_state which calls
        # forward_prefill_export → _forward_prefill_export_impl (PATCHED!)
        out = self.model.model.process_layers_prefill_export_local_state(
            hidden_states=hidden_states,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            kv_cache_0=None,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            linear_conv_state=linear_conv_state,
            linear_recurrent_state=linear_recurrent_state,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            apply_final_norm=False,
            expected_batch_size=1,
            expected_seq_len=self.export_seq_len,
            valid_len=valid_len,
        )
        if self._is_last_chunk:
            out = self.model.model.norm(out)
            seq_len = self.export_seq_len
            positions = torch.arange(seq_len, device=out.device, dtype=torch.int32)
            target = valid_len - 1
            selector = (positions == target).to(out.dtype)
            selector = selector.reshape(1, 1, seq_len)
            out = torch.bmm(selector, out)
        return out, linear_conv_state, linear_recurrent_state

wrapper = PrefillWrapper(model, start_layer, end_layer, SEQ_LEN).eval()
print(f'  Wrapper built: {local_num_layers} layers, seq_len={SEQ_LEN}')
print(f'  Linear conv shape: {wrapper._lin_conv_shape}')
print(f'  Linear rec shape:  {wrapper._lin_rec_shape}')

# ── Trace ────────────────────────────────────────────────────────
print(f'\n{"="*70}')
print('STEP 4: torch.jit.trace (recurrent loop over 512 tokens — may take minutes)')
print('=' * 70)

hidden = torch.zeros((1, SEQ_LEN, cfg.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE)
pos_ids = torch.zeros((SEQ_LEN,), dtype=torch.int32, device=TEST_DEVICE)
mask = torch.zeros((1, 1, SEQ_LEN, CTX_LEN), dtype=MODEL_DTYPE, device=TEST_DEVICE)
cur_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
lin_conv = torch.zeros(wrapper._lin_conv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
lin_rec = torch.zeros(wrapper._lin_rec_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
valid_len = torch.tensor([SEQ_LEN], dtype=torch.int32, device=TEST_DEVICE)

# Reset state buffers
with torch.no_grad():
    wrapper.k_cache.zero_()
    wrapper.v_cache.zero_()

t0 = time.time()
print('  Tracing...')
traced = torch.jit.trace(
    wrapper,
    (hidden, pos_ids, mask, cur_pos, lin_conv, lin_rec, valid_len),
    check_trace=False,
)
trace_time = time.time() - t0
print(f'  Traced in {trace_time:.1f}s')

# Reset caches after trace
with torch.no_grad():
    wrapper.k_cache.zero_()
    wrapper.v_cache.zero_()
    for name, buf in traced.named_buffers():
        if 'cache' in name:
            buf.zero_()

# ── Convert to CoreML ────────────────────────────────────────────
print(f'\n{"="*70}')
print('STEP 5: ct.convert (no LUT quantization — raw fp16)')
print('=' * 70)

t0 = time.time()
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="hidden_states", shape=hidden.shape, dtype=np.float16),
        ct.TensorType(name="position_ids", shape=pos_ids.shape, dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=mask.shape, dtype=np.float16),
        ct.TensorType(name="current_pos", shape=cur_pos.shape, dtype=np.int32),
        ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
        ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
        ct.TensorType(name="valid_len", shape=valid_len.shape, dtype=np.int32),
    ],
    outputs=[
        ct.TensorType(name="output_hidden_states", dtype=np.float16),
        ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
        ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
    ],
    states=wrapper.states,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
    convert_to="mlprogram",
)
convert_time = time.time() - t0
print(f'  Converted in {convert_time:.1f}s')

# ── Save ─────────────────────────────────────────────────────────
out_path = '/Users/yw68/Anemll/qwen3_5_stable_models_6chunk/test_hybrid_chunk0.mlpackage'
mlmodel.save(out_path)
print(f'  Saved to {out_path}')

# ── Load on ANE and run predict ──────────────────────────────────
print(f'\n{"="*70}')
print('STEP 6: Load on ANE (CPU_AND_NE) and run predict()')
print('=' * 70)

t0 = time.time()
loaded = ct.models.MLModel(out_path, compute_units=cu)
load_time = time.time() - t0
print(f'  Loaded on ANE in {load_time:.1f}s')

# Run predict with random data
state = loaded.make_state()
inp = {
    "hidden_states": np.random.randn(1, SEQ_LEN, cfg.hidden_size).astype(np.float16),
    "position_ids": np.arange(SEQ_LEN, dtype=np.int32),
    "causal_mask": np.zeros((1, 1, SEQ_LEN, CTX_LEN), dtype=np.float16),
    "current_pos": np.array([0], dtype=np.int32),
    "linear_conv_state": np.zeros(wrapper._lin_conv_shape, dtype=np.float16),
    "linear_recurrent_state": np.zeros(wrapper._lin_rec_shape, dtype=np.float16),
    "valid_len": np.array([SEQ_LEN], dtype=np.int32),
}

print('  Running predict()...')
t0 = time.time()
out = loaded.predict(inp, state=state)
t_predict = time.time() - t0
print(f'  predict() completed in {t_predict:.3f}s')

h = out['output_hidden_states']
print(f'  Output shape: {h.shape}')
print(f'  Output range: [{h.min():.4f}, {h.max():.4f}]')
print(f'  Output has NaN: {np.any(np.isnan(h))}')
print(f'  Output has Inf: {np.any(np.isinf(h))}')

# ── Benchmark ────────────────────────────────────────────────────
print(f'\n{"="*70}')
print('STEP 7: Latency benchmark (hybrid vs existing prefill)')
print('=' * 70)

WARMUP = 2
TRIALS = 5

# Benchmark hybrid
print(f'\n  Hybrid (force_recurrent=True):')
for w in range(WARMUP):
    state = loaded.make_state()
    loaded.predict(inp, state=state)
    print(f'    warmup {w+1}')

hybrid_times = []
for trial in range(TRIALS):
    state = loaded.make_state()
    t0 = time.perf_counter()
    loaded.predict(inp, state=state)
    t = time.perf_counter() - t0
    hybrid_times.append(t)
    print(f'    trial {trial+1}: {t*1000:.1f} ms')

# Benchmark existing prefill
print(f'\n  Existing prefill (chunk_gated_delta_rule):')
try:
    existing = ct.models.MLModel(EXISTING_CHUNK0, compute_units=cu, function_name='prefill')
    for w in range(WARMUP):
        state = existing.make_state()
        existing.predict(inp, state=state)
        print(f'    warmup {w+1}')

    existing_times = []
    for trial in range(TRIALS):
        state = existing.make_state()
        t0 = time.perf_counter()
        existing.predict(inp, state=state)
        t = time.perf_counter() - t0
        existing_times.append(t)
        print(f'    trial {trial+1}: {t*1000:.1f} ms')
except Exception as e:
    print(f'    SKIP — could not load existing model: {e}')
    existing_times = None

# ── Summary ──────────────────────────────────────────────────────
print(f'\n{"="*70}')
print('RESULTS')
print('=' * 70)
has_nan = np.any(np.isnan(h))
has_inf = np.any(np.isinf(h))

h_avg = np.mean(hybrid_times) * 1000
h_std = np.std(hybrid_times) * 1000

print(f'  Chunk0 ({local_num_layers} layers: {layer_types})')
print(f'  seq_len={SEQ_LEN}, ctx_len={CTX_LEN}')
print()
print(f'  Trace time:   {trace_time:8.1f}s')
print(f'  Convert time: {convert_time:8.1f}s')
print(f'  ANE load:     {load_time:8.1f}s')
print()
print(f'  Hybrid predict latency: {h_avg:8.1f} ± {h_std:.1f} ms')

if existing_times is not None:
    e_avg = np.mean(existing_times) * 1000
    e_std = np.std(existing_times) * 1000
    ratio = h_avg / e_avg
    print(f'  Existing predict latency: {e_avg:8.1f} ± {e_std:.1f} ms')
    print(f'  Hybrid / Existing ratio:  {ratio:8.2f}×')

print()
print(f'  ANE LOADABLE:  {"YES" if load_time > 0 else "FAILED"}')
print(f'  predict() OK:  {"YES" if t_predict > 0 else "FAILED"}')
print(f'  No NaN:        {"YES" if not has_nan else "FAILED"}')
print(f'  No Inf:        {"YES" if not has_inf else "FAILED"}')
