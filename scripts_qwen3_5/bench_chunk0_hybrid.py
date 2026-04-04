#!/usr/bin/env python3
"""Benchmark chunk0: batched vs sequential vs hybrid (full batched + linear sequential).

Creates chunk0's 6 layers (5 linear + 1 full) with random weights and measures
three prefill strategies on CPU. Reports relative timings.

Chunk0 layers: [linear, linear, linear, full, linear, linear]
  - "All batched":     all 6 layers process seq_len tokens at once (current prefill)
  - "All sequential":  all 6 layers process 1 token at a time in a loop
  - "Hybrid":          full_attention layers batched, linear_attention layers sequential
                       (= what force_recurrent=True re-export would produce)
"""
import sys, os, time, json
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anemll.models.qwen3_5_model import (
    Qwen35Config, Qwen35DecoderLayer, Qwen35RMSNorm, Qwen35MLP,
    Qwen35FullAttention, Qwen35LinearAttention, MODEL_DTYPE,
    ane_conv_state_shape,
)

# ── Load config ──────────────────────────────────────────────────
cfg_path = '/Users/yw68/Anemll/models/Qwen__Qwen3.5-4B/config.json'
with open(cfg_path) as f:
    raw = json.load(f)
cfg = Qwen35Config(raw)
layer_types = cfg.text_config.layer_types

# Chunk0: layers 0-5
START, END = 0, 6
chunk0_types = layer_types[START:END]
print(f'Chunk0 layers {START}-{END-1}: {chunk0_types}')
print(f'  Full attention:   {sum(1 for t in chunk0_types if t == "full_attention")}')
print(f'  Linear attention: {sum(1 for t in chunk0_types if t == "linear_attention")}')

# ── Build chunk0 layers (random weights) ─────────────────────────
print('\nBuilding chunk0 layers with random weights...')
torch.manual_seed(42)
layers = []
for i, lt in enumerate(chunk0_types):
    layer = Qwen35DecoderLayer(cfg, layer_type=lt)
    layers.append(layer)
    print(f'  layer {i} ({lt}): built')

final_norm = Qwen35RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

# Test seq lengths
SEQ_LENS = [64, 128, 256, 512]
WARMUP = 3
TRIALS = 5
CTX_LEN = cfg.state_length  # for KV cache

print(f'\nConfig: hidden={cfg.hidden_size}, heads={cfg.num_attention_heads}, '
      f'kv_heads={cfg.num_key_value_heads}, head_dim={cfg.head_dim}')
print(f'Linear: k_heads={cfg.text_config.linear_num_key_heads}, '
      f'v_heads={cfg.text_config.linear_num_value_heads}, '
      f'k_dim={cfg.text_config.linear_key_head_dim}, '
      f'v_dim={cfg.text_config.linear_value_head_dim}, '
      f'conv_kernel={cfg.text_config.linear_conv_kernel_dim}')


# ── Helper: build masks/states ───────────────────────────────────
def build_prefill_causal_mask(seq_len, ctx_len):
    """[1, 1, seq_len, ctx_len] causal mask."""
    mask = torch.full((1, 1, seq_len, ctx_len), -65504.0, dtype=MODEL_DTYPE)
    for i in range(seq_len):
        mask[0, 0, i, :i+1] = 0
    return mask

def build_decode_causal_mask(pos, ctx_len):
    """[1, 1, 1, ctx_len] decode mask at position pos."""
    mask = torch.full((1, 1, 1, ctx_len), -65504.0, dtype=MODEL_DTYPE)
    mask[0, 0, 0, :pos+1] = 0
    return mask

def make_linear_states(n_layers):
    """Fresh zero conv_state and rec_state for n linear layers."""
    la = layers[0].self_attn  # grab a linear attention instance for dims
    conv_dim = la.conv_dim
    conv_kernel = la.linear_conv_kernel_dim
    n_v_heads = la.num_v_heads
    k_dim = la.head_k_dim
    v_dim = la.head_v_dim
    conv_states = [torch.zeros(1, conv_dim, conv_kernel, dtype=MODEL_DTYPE) for _ in range(n_layers)]
    rec_states = [torch.zeros(1, n_v_heads, k_dim, v_dim, dtype=MODEL_DTYPE) for _ in range(n_layers)]
    return conv_states, rec_states

def make_kv_cache(n_full_layers):
    """Fresh KV cache for full attention layers."""
    return [
        (torch.zeros(cfg.num_key_value_heads, CTX_LEN, cfg.head_dim, dtype=MODEL_DTYPE),
         torch.zeros(cfg.num_key_value_heads, CTX_LEN, cfg.head_dim, dtype=MODEL_DTYPE))
        for _ in range(n_full_layers)
    ]


# ══════════════════════════════════════════════════════════════════
# MODE A: All batched (current prefill path)
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_all_batched(seq_len):
    hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=MODEL_DTYPE)
    mask = build_prefill_causal_mask(seq_len, CTX_LEN)
    pos_ids = torch.arange(seq_len, dtype=torch.long)
    kv_caches = make_kv_cache(sum(1 for t in chunk0_types if t == 'full_attention'))
    lin_convs, lin_recs = make_linear_states(sum(1 for t in chunk0_types if t == 'linear_attention'))

    full_idx, lin_idx = 0, 0
    for i, layer in enumerate(layers):
        if layer.layer_type == 'linear_attention':
            x = layer.input_layernorm(hidden)
            attn_out, next_conv, next_rec = layer.self_attn.forward_prefill(
                hidden_states=x,
                conv_state=lin_convs[lin_idx],
                recurrent_state=lin_recs[lin_idx],
                has_previous_state=False,
                expected_batch_size=1,
                expected_seq_len=seq_len,
            )
            lin_convs[lin_idx] = next_conv
            lin_recs[lin_idx] = next_rec
            lin_idx += 1
            hidden = hidden + attn_out
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        else:
            x = layer.input_layernorm(hidden)
            q, k, v, gate = layer.self_attn.get_new_kv_cache_prefill(x, pos_ids)
            kc, vc = kv_caches[full_idx]
            kc[:, :seq_len, :] = k.squeeze(0)
            vc[:, :seq_len, :] = v.squeeze(0)
            attn_out = layer.self_attn.forward_prefill(
                hidden_states=x, query_states=q,
                kv_cache_layer=(kc, vc), causal_mask=mask, gate=gate,
            )
            full_idx += 1
            hidden = hidden + attn_out
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden


# ══════════════════════════════════════════════════════════════════
# MODE B: All sequential (current workaround)
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_all_sequential(seq_len):
    # Pre-embed all tokens
    all_hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=MODEL_DTYPE)
    kv_caches = make_kv_cache(sum(1 for t in chunk0_types if t == 'full_attention'))
    lin_convs, lin_recs = make_linear_states(sum(1 for t in chunk0_types if t == 'linear_attention'))

    for t in range(seq_len):
        hidden = all_hidden[:, t:t+1, :]  # [1, 1, H]
        mask = build_decode_causal_mask(t, CTX_LEN)
        pos_arr = torch.tensor([t], dtype=torch.long)

        full_idx, lin_idx = 0, 0
        for i, layer in enumerate(layers):
            if layer.layer_type == 'linear_attention':
                x = layer.input_layernorm(hidden)
                attn_out, next_conv, next_rec = layer.self_attn.forward_regular(
                    hidden_states=x,
                    conv_state=lin_convs[lin_idx],
                    recurrent_state=lin_recs[lin_idx],
                    has_previous_state=(t > 0),
                    expected_batch_size=1,
                    expected_seq_len=1,
                    force_recurrent=True,
                )
                lin_convs[lin_idx] = next_conv
                lin_recs[lin_idx] = next_rec
                lin_idx += 1
                hidden = hidden + attn_out
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            else:
                x = layer.input_layernorm(hidden)
                q, k, v, gate = layer.self_attn.get_new_kv_cache(x, pos_arr)
                kc, vc = kv_caches[full_idx]
                kc[:, t:t+1, :] = k.squeeze(0)
                vc[:, t:t+1, :] = v.squeeze(0)
                attn_out = layer.self_attn.forward_regular(
                    hidden_states=x, query_states=q,
                    kv_cache_layer=(kc, vc), causal_mask=mask, gate=gate,
                )
                full_idx += 1
                hidden = hidden + attn_out
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        all_hidden[:, t:t+1, :] = hidden
    return all_hidden


# ══════════════════════════════════════════════════════════════════
# MODE C: Hybrid — full_attention batched, linear_attention sequential
# (This is what force_recurrent=True re-export achieves)
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_hybrid(seq_len):
    """Full attention layers see all tokens at once;
    linear attention layers process one token at a time."""
    all_hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=MODEL_DTYPE)
    kv_caches = make_kv_cache(sum(1 for t in chunk0_types if t == 'full_attention'))
    lin_convs, lin_recs = make_linear_states(sum(1 for t in chunk0_types if t == 'linear_attention'))

    # We need to interleave batched and sequential processing.
    # Process layers in order. When we hit a linear layer, we must loop.
    # When we hit a full layer, we process the full sequence.
    #
    # The complication: layers are interleaved (lin, lin, lin, full, lin, lin).
    # Between each set of linear layers, we accumulate the full hidden sequence,
    # then the full attention layer processes it all at once.

    # Strategy: process each layer one at a time.
    # For linear layers: loop over tokens
    # For full layers: process all tokens at once

    full_idx, lin_idx = 0, 0
    for i, layer in enumerate(layers):
        if layer.layer_type == 'linear_attention':
            # Process tokens one at a time through this linear layer
            for t in range(seq_len):
                hidden_t = all_hidden[:, t:t+1, :]  # [1, 1, H]
                x = layer.input_layernorm(hidden_t)
                attn_out, next_conv, next_rec = layer.self_attn.forward_regular(
                    hidden_states=x,
                    conv_state=lin_convs[lin_idx],
                    recurrent_state=lin_recs[lin_idx],
                    has_previous_state=(t > 0),
                    expected_batch_size=1,
                    expected_seq_len=1,
                    force_recurrent=True,
                )
                lin_convs[lin_idx] = next_conv
                lin_recs[lin_idx] = next_rec
                hidden_t = hidden_t + attn_out
                hidden_t = hidden_t + layer.mlp(layer.post_attention_layernorm(hidden_t))
                all_hidden[:, t:t+1, :] = hidden_t
            lin_idx += 1
        else:
            # Process all tokens at once through this full attention layer
            hidden = all_hidden  # [1, seq_len, H]
            mask = build_prefill_causal_mask(seq_len, CTX_LEN)
            pos_ids = torch.arange(seq_len, dtype=torch.long)
            x = layer.input_layernorm(hidden)
            q, k, v, gate = layer.self_attn.get_new_kv_cache_prefill(x, pos_ids)
            kc, vc = kv_caches[full_idx]
            kc[:, :seq_len, :] = k.squeeze(0)
            vc[:, :seq_len, :] = v.squeeze(0)
            attn_out = layer.self_attn.forward_prefill(
                hidden_states=x, query_states=q,
                kv_cache_layer=(kc, vc), causal_mask=mask, gate=gate,
            )
            full_idx += 1
            all_hidden = hidden + attn_out
            all_hidden = all_hidden + layer.mlp(layer.post_attention_layernorm(all_hidden))
    return all_hidden


# ══════════════════════════════════════════════════════════════════
# Also measure per-layer-type times
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def time_single_linear_batched(seq_len):
    """Time one linear attention layer processing seq_len tokens at once."""
    la = layers[0]  # first layer is linear
    hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=MODEL_DTYPE)
    conv_dim = la.self_attn.conv_dim
    conv_kernel = la.self_attn.linear_conv_kernel_dim
    conv_state = torch.zeros(1, conv_dim, conv_kernel, dtype=MODEL_DTYPE)
    rec_state = torch.zeros(1, la.self_attn.num_v_heads, la.self_attn.head_k_dim, la.self_attn.head_v_dim, dtype=MODEL_DTYPE)

    x = la.input_layernorm(hidden)
    t0 = time.perf_counter()
    attn_out, _, _ = la.self_attn.forward_prefill(
        hidden_states=x, conv_state=conv_state, recurrent_state=rec_state,
        has_previous_state=False, expected_batch_size=1, expected_seq_len=seq_len,
    )
    return time.perf_counter() - t0

@torch.no_grad()
def time_single_linear_sequential(seq_len):
    """Time one linear attention layer processing seq_len tokens sequentially."""
    la = layers[0]
    hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=MODEL_DTYPE)
    conv_dim = la.self_attn.conv_dim
    conv_kernel = la.self_attn.linear_conv_kernel_dim
    conv_state = torch.zeros(1, conv_dim, conv_kernel, dtype=MODEL_DTYPE)
    rec_state = torch.zeros(1, la.self_attn.num_v_heads, la.self_attn.head_k_dim, la.self_attn.head_v_dim, dtype=MODEL_DTYPE)

    all_x = la.input_layernorm(hidden)
    t0 = time.perf_counter()
    for t in range(seq_len):
        x_t = all_x[:, t:t+1, :]
        _, conv_state, rec_state = la.self_attn.forward_regular(
            hidden_states=x_t, conv_state=conv_state, recurrent_state=rec_state,
            has_previous_state=(t > 0), expected_batch_size=1, expected_seq_len=1,
            force_recurrent=True,
        )
    return time.perf_counter() - t0

@torch.no_grad()
def time_single_full_batched(seq_len):
    """Time one full attention layer processing seq_len tokens at once."""
    # Find the full attention layer
    fa_layer = [l for l in layers if l.layer_type == 'full_attention'][0]
    hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=MODEL_DTYPE)
    mask = build_prefill_causal_mask(seq_len, CTX_LEN)
    pos_ids = torch.arange(seq_len, dtype=torch.long)
    kc = torch.zeros(cfg.num_key_value_heads, CTX_LEN, cfg.head_dim, dtype=MODEL_DTYPE)
    vc = torch.zeros(cfg.num_key_value_heads, CTX_LEN, cfg.head_dim, dtype=MODEL_DTYPE)

    x = fa_layer.input_layernorm(hidden)
    q, k, v, gate = fa_layer.self_attn.get_new_kv_cache_prefill(x, pos_ids)
    kc[:, :seq_len, :] = k.squeeze(0)
    vc[:, :seq_len, :] = v.squeeze(0)
    t0 = time.perf_counter()
    fa_layer.self_attn.forward_prefill(
        hidden_states=x, query_states=q,
        kv_cache_layer=(kc, vc), causal_mask=mask, gate=gate,
    )
    return time.perf_counter() - t0


# ══════════════════════════════════════════════════════════════════
# Run all benchmarks
# ══════════════════════════════════════════════════════════════════
print(f'\n{"="*80}')
print('CHUNK0 LATENCY BENCHMARK (PyTorch CPU, random weights)')
print(f'Layers: {chunk0_types}')
print(f'Warmup={WARMUP}, Trials={TRIALS}')
print(f'{"="*80}')

all_results = []

for seq_len in SEQ_LENS:
    print(f'\n{"─"*80}')
    print(f'seq_len = {seq_len}')
    print(f'{"─"*80}')

    # ── Per-layer-type timings ───────────────────────────────────
    print(f'\n  Per-layer-type (single layer):')
    for _ in range(WARMUP):
        time_single_linear_batched(seq_len)
        time_single_linear_sequential(seq_len)
        time_single_full_batched(seq_len)

    lb_times = [time_single_linear_batched(seq_len) for _ in range(TRIALS)]
    ls_times = [time_single_linear_sequential(seq_len) for _ in range(TRIALS)]
    fb_times = [time_single_full_batched(seq_len) for _ in range(TRIALS)]
    lb = sum(lb_times)/len(lb_times) * 1000
    ls = sum(ls_times)/len(ls_times) * 1000
    fb = sum(fb_times)/len(fb_times) * 1000
    print(f'    Linear batched (chunk_gated):     {lb:8.1f} ms')
    print(f'    Linear sequential (recurrent):    {ls:8.1f} ms')
    print(f'    Linear seq/batch ratio:           {ls/lb:8.1f}×')
    print(f'    Full attention batched:            {fb:8.1f} ms')

    # ── Mode A: all batched ──────────────────────────────────────
    print(f'\n  Mode A: all batched (current prefill):')
    for _ in range(WARMUP):
        run_all_batched(seq_len)
    a_times = []
    for trial in range(TRIALS):
        t0 = time.perf_counter()
        run_all_batched(seq_len)
        t = time.perf_counter() - t0
        a_times.append(t)
        print(f'    trial {trial+1}: {t*1000:.1f} ms')
    a_avg = sum(a_times)/len(a_times) * 1000

    # ── Mode B: all sequential ───────────────────────────────────
    print(f'\n  Mode B: all sequential (current workaround):')
    for _ in range(WARMUP):
        run_all_sequential(seq_len)
    b_times = []
    for trial in range(TRIALS):
        t0 = time.perf_counter()
        run_all_sequential(seq_len)
        t = time.perf_counter() - t0
        b_times.append(t)
        print(f'    trial {trial+1}: {t*1000:.1f} ms')
    b_avg = sum(b_times)/len(b_times) * 1000

    # ── Mode C: hybrid ───────────────────────────────────────────
    print(f'\n  Mode C: hybrid (full batched + linear sequential):')
    for _ in range(WARMUP):
        run_hybrid(seq_len)
    c_times = []
    for trial in range(TRIALS):
        t0 = time.perf_counter()
        run_hybrid(seq_len)
        t = time.perf_counter() - t0
        c_times.append(t)
        print(f'    trial {trial+1}: {t*1000:.1f} ms')
    c_avg = sum(c_times)/len(c_times) * 1000

    all_results.append({
        'seq_len': seq_len,
        'a_batched': a_avg,
        'b_sequential': b_avg,
        'c_hybrid': c_avg,
        'lin_batch': lb,
        'lin_seq': ls,
        'full_batch': fb,
    })

    print(f'\n  Summary for seq_len={seq_len}:')
    print(f'    A (all batched):     {a_avg:8.1f} ms')
    print(f'    B (all sequential):  {b_avg:8.1f} ms  ({b_avg/a_avg:.1f}× vs A)')
    print(f'    C (hybrid):          {c_avg:8.1f} ms  ({c_avg/a_avg:.1f}× vs A)')


# ══════════════════════════════════════════════════════════════════
# Final Summary
# ══════════════════════════════════════════════════════════════════
print(f'\n\n{"="*80}')
print('FINAL SUMMARY — Chunk0 (5 linear + 1 full attention)')
print(f'{"="*80}')
print(f'\n{"seq":>5s}  {"A:batched":>11s}  {"B:sequential":>13s}  {"C:hybrid":>10s}  {"B/A":>6s}  {"C/A":>6s}  {"C/B":>6s}')
print(f'{"-"*63}')
for r in all_results:
    ba = r['b_sequential'] / r['a_batched']
    ca = r['c_hybrid'] / r['a_batched']
    cb = r['c_hybrid'] / r['b_sequential']
    print(f'{r["seq_len"]:>5d}  {r["a_batched"]:>9.1f}ms  {r["b_sequential"]:>11.1f}ms  {r["c_hybrid"]:>8.1f}ms  '
          f'{ba:>5.1f}×  {ca:>5.1f}×  {cb:>5.1f}×')

print(f'\nPer-layer breakdown:')
print(f'{"seq":>5s}  {"lin_batch":>11s}  {"lin_seq":>9s}  {"lin_s/b":>8s}  {"full_batch":>11s}')
print(f'{"-"*50}')
for r in all_results:
    print(f'{r["seq_len"]:>5d}  {r["lin_batch"]:>9.1f}ms  {r["lin_seq"]:>7.1f}ms  '
          f'{r["lin_seq"]/r["lin_batch"]:>7.1f}×  {r["full_batch"]:>9.1f}ms')

print(f'''
KEY:
  A = All batched (current prefill with _chunk_gated_delta_rule)
  B = All sequential (current PREFILL_CROSSOVER=999999 workaround)
  C = Hybrid (full_attn batched + linear_attn sequential via force_recurrent=True)

  C/A tells you the slowdown of the proposed re-export vs current batch prefill.
  C/B tells you the speedup of the proposed re-export vs current sequential workaround.
''')
