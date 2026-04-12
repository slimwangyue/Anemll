#!/usr/bin/env python3
"""
ANE Rewrite Investigation for FLLL Decode Chunks

Diagnoses exact ANE-hostile ops, implements rewrites, and measures placement impact.

Phases:
  1. Deep MIL diagnosis — find every hostile op with full context
  2. Source-to-MIL tracing — map MIL ops back to Python source
  3. Implement rewrites and export new chunks
  4. Measure ANE/CPU placement before vs after
  5. Correctness validation

Usage:
  python tests/dev/ane_rewrite_investigation.py [--phase N] [--chunk N]
"""
import sys, os, time, gc, re, argparse, json, warnings
from collections import Counter, defaultdict
import numpy as np

REPO_ROOT = '/Volumes/MySSD/Anemll'
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts_qwen3_5'))
os.chdir(REPO_ROOT)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ.setdefault('TMPDIR', '/Volumes/MySSD/tmp')

import torch
torch.set_grad_enabled(False)
import coremltools as ct

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

HF_MODEL = os.path.join(REPO_ROOT, 'models', 'Qwen__Qwen3.5-4B')
MODEL_DIR = os.path.join(REPO_ROOT, 'qwen3_5_stable_lut4ffn_lut6em_fp32')
COMBINED_DIR = os.path.join(MODEL_DIR, 'combined_LUT4_dedup')
ARTIFACT_DIR = os.path.join(REPO_ROOT, 'artifacts', 'ane_rewrite_investigation')
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DTYPE_MAP = {1: 'fp32', 10: 'fp16', 11: 'fp16', 5: 'int32', 7: 'int64', 21: 'int16'}
ANE_HOSTILE_OPS = frozenset({
    'gather', 'scatter', 'scatter_along_axis', 'one_hot',
    'greater_equal', 'less', 'greater', 'less_equal', 'equal', 'not_equal',
    'select', 'where', 'logical_and', 'logical_or', 'logical_not',
    'read_state', 'coreml_update_state',
})

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Deep MIL Diagnosis
# ═══════════════════════════════════════════════════════════════════════

def deep_mil_diagnosis(model_path, fn_name="infer"):
    """Find every ANE-hostile op with full I/O context."""
    spec = ct.utils.load_spec(model_path)
    if not hasattr(spec, 'mlProgram'):
        return []

    prog = spec.mlProgram
    if fn_name not in prog.functions:
        print(f"  WARNING: function '{fn_name}' not found, available: {list(prog.functions.keys())}")
        return []

    fn = prog.functions[fn_name]
    hostile_ops = []
    all_ops = []

    for _bname, block in fn.block_specializations.items():
        for idx, op in enumerate(block.operations):
            ot = op.type
            # Get output info
            out_name = op.outputs[0].name if op.outputs else "?"
            try:
                out_dtype = DTYPE_MAP.get(op.outputs[0].type.tensorType.dataType, '?')
                dims = list(op.outputs[0].type.tensorType.dimensions)
                out_shape = [d.constant.size for d in dims] if dims else []
            except Exception:
                out_dtype = '?'
                out_shape = []

            # Get input info
            inputs_info = {}
            for inp_name in op.inputs:
                inp_val = op.inputs[inp_name]
                # Try to get the binding name
                try:
                    if inp_val.HasField('name'):
                        inputs_info[inp_name] = {'type': 'ref', 'name': inp_val.name}
                    elif inp_val.HasField('value'):
                        ival = inp_val.value
                        if ival.HasField('immediateValue'):
                            imm = ival.immediateValue
                            if imm.HasField('tensor'):
                                inputs_info[inp_name] = {'type': 'const_tensor'}
                            elif imm.HasField('scalar'):
                                inputs_info[inp_name] = {'type': 'scalar'}
                            else:
                                inputs_info[inp_name] = {'type': 'immediate'}
                        else:
                            inputs_info[inp_name] = {'type': 'value'}
                    else:
                        inputs_info[inp_name] = {'type': 'unknown'}
                except Exception:
                    inputs_info[inp_name] = {'type': 'parse_error'}

            record = {
                'idx': idx,
                'type': ot,
                'out_name': out_name,
                'out_dtype': out_dtype,
                'out_shape': out_shape,
                'inputs': inputs_info,
            }
            all_ops.append(record)

            if ot in ANE_HOSTILE_OPS:
                hostile_ops.append(record)

    return hostile_ops, all_ops


def find_op_neighborhood(all_ops, hostile_idx, window=5):
    """Find ops near a hostile op to understand dataflow context."""
    start = max(0, hostile_idx - window)
    end = min(len(all_ops), hostile_idx + window + 1)
    return all_ops[start:end]


def run_phase1(chunk_idx=1):
    """Phase 1: Deep MIL diagnosis of hostile ops."""
    print("=" * 70)
    print("  PHASE 1: Deep MIL Diagnosis of ANE-Hostile Ops")
    print("=" * 70)

    model_path = os.path.join(COMBINED_DIR, f'chunk{chunk_idx}.mlpackage')
    if not os.path.exists(model_path):
        print(f"  ERROR: {model_path} not found")
        return None

    results = {}
    for fn_name in ['infer', 'prefill']:
        print(f"\n  ── Function: {fn_name} (Chunk {chunk_idx}) ──")
        hostile_ops, all_ops = deep_mil_diagnosis(model_path, fn_name)
        results[fn_name] = {'hostile': hostile_ops, 'all_ops': all_ops}

        if not hostile_ops:
            print(f"    No ANE-hostile ops found!")
            continue

        print(f"    Total ops: {len(all_ops)}, Hostile: {len(hostile_ops)}")
        type_counts = Counter(h['type'] for h in hostile_ops)
        print(f"    Hostile breakdown: {dict(type_counts)}")

        print(f"\n    Hostile ops detail:")
        for h in hostile_ops:
            print(f"      [{h['idx']:4d}] {h['type']:25s} → {h['out_name']}")
            print(f"             shape={h['out_shape']}  dtype={h['out_dtype']}")
            for inp_name, inp_info in h['inputs'].items():
                print(f"             input.{inp_name}: {inp_info}")

            # Show neighborhood
            print(f"             ─── Neighborhood (±3 ops) ───")
            neighbors = find_op_neighborhood(all_ops, h['idx'], window=3)
            for n in neighbors:
                marker = " ►►►" if n['idx'] == h['idx'] else "    "
                print(f"        {marker} [{n['idx']:4d}] {n['type']:20s} → {n['out_name'][:50]}")

        print()

    return results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Source identification - Convert with debug tracing
# ═══════════════════════════════════════════════════════════════════════

def run_phase2():
    """Phase 2: Export chunk with debug tracing to map ops to source."""
    print("=" * 70)
    print("  PHASE 2: Source Tracing via Debug Conversion")
    print("=" * 70)

    from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE

    print("\n  Loading model weights...")
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Convert chunk 1 (FLLL) decode with standard code
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    chunk_idx = 1
    start_layer, end_layer = CHUNK_RANGES[chunk_idx]
    print(f"\n  Converting chunk {chunk_idx} (layers {start_layer}-{end_layer-1}) baseline...")

    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
        compute_precision="float32",
    )

    t0 = time.time()
    ml = conv.convert_part_2(
        model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS,
        override_start_layer=start_layer, override_end_layer=end_layer,
    )
    baseline_path = os.path.join(ARTIFACT_DIR, f'baseline_chunk{chunk_idx}_decode.mlpackage')
    ml.save(baseline_path)
    print(f"  Baseline saved in {time.time()-t0:.1f}s: {baseline_path}")
    del ml; gc.collect()

    # Run MIL diagnosis on baseline
    print(f"\n  Diagnosing baseline hostile ops...")
    hostile_ops, all_ops = deep_mil_diagnosis(baseline_path, "main")
    if not hostile_ops:
        # Try without function name (single-function model)
        spec = ct.utils.load_spec(baseline_path)
        fns = list(spec.mlProgram.functions.keys())
        print(f"  Available functions: {fns}")
        for fn in fns:
            hostile_ops, all_ops = deep_mil_diagnosis(baseline_path, fn)
            if hostile_ops:
                break

    if hostile_ops:
        print(f"\n  Found {len(hostile_ops)} hostile ops in baseline:")
        for h in hostile_ops:
            print(f"    [{h['idx']:4d}] {h['type']:25s} → {h['out_name']}")
            # Show 5 ops before
            start = max(0, h['idx'] - 5)
            for n in all_ops[start:h['idx']]:
                print(f"           [{n['idx']:4d}] {n['type']:20s} → {n['out_name'][:60]}")
    else:
        print(f"  No hostile ops found in baseline decode model")
        print(f"  Total ops: {len(all_ops)}")
        # Show all op types
        type_counts = Counter(op['type'] for op in all_ops)
        print(f"  Op types: {dict(type_counts.most_common(30))}")

    return model, cfg, hostile_ops, all_ops


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Implement Rewrites
# ═══════════════════════════════════════════════════════════════════════

def make_onthefly_rope_get(orig_rotary):
    """Monkey-patch RoPE to compute cos/sin on-the-fly instead of table lookup.

    Replaces:
      cos = self.cos_cached[:, pos_ids]   # gather op
      sin = self.sin_cached[:, pos_ids]   # gather op

    With ANE-friendly:
      freqs = pos_float * inv_freq        # elementwise mul
      cos = cat(freqs.cos(), freqs.cos()) # elementwise cos + cat
      sin = cat(freqs.sin(), freqs.sin()) # elementwise sin + cat
    """
    inv_freq = orig_rotary.inv_freq  # [rotary_dim/2]

    def get_onthefly(self, x, position_ids):
        if position_ids.dim() > 1:
            pos_ids = position_ids.squeeze(0)
        else:
            pos_ids = position_ids
        # pos_ids: [S] int32/int64 → float for computation
        t = pos_ids.float().unsqueeze(-1)  # [S, 1]
        freqs = t * inv_freq.unsqueeze(0)   # [S, rotary_dim/2] — elementwise mul
        emb = torch.cat([freqs, freqs], dim=-1)  # [S, rotary_dim]
        cos = emb.cos().unsqueeze(0).to(x.dtype)  # [1, S, rotary_dim]
        sin = emb.sin().unsqueeze(0).to(x.dtype)  # [1, S, rotary_dim]
        return cos, sin

    return get_onthefly


def make_arithmetic_softplus():
    """Replace F.relu + abs (which may compile to comparison ops) with
    pure arithmetic softplus: log(1 + exp(x)) = x + log(1 + exp(-x))
    using only elementwise ops.

    The stable softplus F.relu(x) + log(1 + exp(-abs(x))) uses:
      - F.relu → may compile to max(x, 0) → greater_equal + select in MIL
      - abs → may compile to select(x >= 0, x, -x) → greater_equal + select

    Alternative: softplus(x) = x/2 + sqrt(x^2 + eps)/2 ... no that's not right
    Actually: softplus(x) = log(1 + exp(x))
    For numerical stability at large x: softplus(x) ≈ x
    For numerical stability at small x: softplus(x) ≈ exp(x)

    Pure arithmetic approach using sigmoid:
      softplus(x) = x * sigmoid(x) + log(1 + exp(-x * sigmoid(x)))
      ... too complex

    Simplest: use torch.nn.functional.softplus which CoreML may map to
    a native op, or use the log-sum-exp trick:
      softplus(x) = log(exp(0) + exp(x)) = logsumexp(0, x)

    Actually the cleanest rewrite:
      softplus(x) = x + log(sigmoid(-x) + 1 - sigmoid(-x) + exp(-x))
      ... still messy

    Best rewrite: just use log(1 + exp(x)) with clamp for stability:
      softplus(x) = where(x > 20, x, log(1 + exp(x)))
    But 'where' is also hostile!

    Arithmetic-only softplus:
      softplus(x) = 0.5 * (x + sqrt(x*x + 4)) ... WRONG, that's softabs

    Actually, the best ANE-native approach is:
      softplus(x) = log(1 + exp(x))
    With overflow handled by:
      softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    The issue is max() and abs().

    ANE HAS native relu and abs as MIL ops. The question is whether they
    compile to greater_equal+select or to native ANE ops.

    Let me try a completely different formulation using sigmoid:
      softplus(x) = integral of sigmoid = x * sigmoid(x) + ... no

    Actually: softplus(x) = -log(sigmoid(-x))
    This uses only sigmoid (native), neg, and log (native). ALL ANE-friendly!
    sigmoid(-x) = 1/(1+exp(x)) → stable for all x
    -log(sigmoid(-x)) = -log(1/(1+exp(x))) = log(1+exp(x))
    Numerically: sigmoid(-x) is always in (0,1), so log is well-defined.
    For large positive x: sigmoid(-x) → 0, log(0) → -inf → clamp needed
    For large negative x: sigmoid(-x) → 1, -log(1) → 0 ✓
    For moderate x: stable ✓

    To handle large positive x: add tiny eps to sigmoid
      softplus(x) = -log(sigmoid(-x) + 1e-7)  ... NO, that changes the function

    Better: softplus(x) = -log(sigmoid(-x))
    For x > 0: sigmoid(-x) = exp(-x)/(1+exp(-x)) > 0 always
    For x = 88 (fp32 max safe): sigmoid(-88) ≈ 6e-39, -log(6e-39) ≈ 88 ✓
    For fp16: x = 11 is about the limit. sigmoid(-11) ≈ 1.6e-5,
              -log(1.6e-5) ≈ 11.04 ✓
    For x = 65: sigmoid(-65) → denorm/0 in fp16 → log(0) = -inf ✗

    So for fp16 we need a stability guard. Alternative:
      softplus(x) = x + log(sigmoid(-x) * exp(-x) + 1) ... circular

    Actually the REAL cleanest:
      softplus(x) = x + F.softplus(-x)  (where F.softplus might be native)
    softplus(x) = x + log(1 + exp(-x))
    For x >> 0: exp(-x) → 0, log(1) → 0, result ≈ x ✓
    For x << 0: x + log(1 + exp(-x)) → x + (-x) → 0... wait that's wrong
    x << 0: exp(-x) is huge, log(1+exp(-x)) ≈ -x, so x + (-x) = 0...
    but softplus(-100) should be ≈ 0, and x + log(1+exp(-x)) = -100 + 100 = 0 ✓
    hmm actually for x = -100: exp(100) overflows in fp16. Problem!

    OK let me just go with:
      softplus(x) = F.softplus(x)
    And see if CoreML maps it to a single native op vs greater_equal+select.
    If not, try: softplus = -log(sigmoid(-x)) with fp32 math (which the code already uses).
    """
    pass  # We'll implement specific alternatives below


def create_rewrite_a_model(model, cfg, chunk_idx):
    """Rewrite A: On-the-fly RoPE (eliminates gather from RoPE table lookup)."""
    from anemll.models.qwen3_5_model import (
        Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
        Qwen35FullAttention
    )
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter, ane_conv_state_shape

    start_layer, end_layer = CHUNK_RANGES[chunk_idx]
    local_num_layers = end_layer - start_layer

    class FFNWrapperRopeRewrite(torch.nn.Module):
        """FFN wrapper with on-the-fly RoPE computation."""
        def __init__(self, model, start_layer, end_layer):
            super().__init__()
            self.model = model
            self.start_layer = start_layer
            self.end_layer = end_layer
            self.local_num_layers = end_layer - start_layer
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
                self._lin_rec_shape = (self.local_num_layers,
                                       cfg.text_config.linear_num_value_heads,
                                       cfg.text_config.linear_key_head_dim,
                                       cfg.text_config.linear_value_head_dim)
                self._has_linear = True
            else:
                self._has_linear = False
            self.states = Qwen35Converter.GetChunkLocalTransformerStates(
                model, self.local_num_layers, prefix="", split_full_attention_kv=True)

            # Pre-compute RoPE inv_freq for on-the-fly computation
            for layer_idx in range(start_layer, end_layer):
                layer = model.model.layers[layer_idx]
                if layer.layer_type == "full_attention":
                    self.register_buffer("rope_inv_freq",
                                        layer.self_attn.rotary.inv_freq.clone())
                    self._rotary_dim = layer.self_attn.rotary.rotary_dim
                    break

        def _rope_onthefly(self, position_ids, dtype, device):
            """Compute RoPE cos/sin on-the-fly. No gather needed."""
            if position_ids.dim() > 1:
                pos_ids = position_ids.squeeze(0)
            else:
                pos_ids = position_ids
            t = pos_ids.to(dtype=torch.float32).unsqueeze(-1)  # [S, 1]
            freqs = t * self.rope_inv_freq.unsqueeze(0)  # [S, rotary_dim/2]
            emb = torch.cat([freqs, freqs], dim=-1)  # [S, rotary_dim]
            cos = emb.cos().unsqueeze(0).to(dtype)  # [1, S, rotary_dim]
            sin = emb.sin().unsqueeze(0).to(dtype)  # [1, S, rotary_dim]
            return cos, sin

        def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                    linear_conv_state, linear_recurrent_state):
            # Process each layer, using on-the-fly RoPE for full-attention layers
            for local_idx, layer_idx in enumerate(range(self.start_layer, self.end_layer)):
                layer = self.model.model.layers[layer_idx]
                if layer.layer_type == "linear_attention":
                    # F-layer: standard path (no RoPE involved)
                    x = layer.input_layernorm(hidden_states)
                    conv_dim = layer.self_attn.conv_dim
                    conv_kernel = layer.self_attn.linear_conv_kernel_dim
                    conv_state_flat = linear_conv_state[local_idx : local_idx + 1]
                    conv_state = conv_state_flat.reshape(1, conv_dim, conv_kernel)
                    recurrent_state = linear_recurrent_state[local_idx : local_idx + 1]
                    attn_out, next_conv, next_rec = layer.self_attn.forward_regular(
                        hidden_states=x, conv_state=conv_state,
                        recurrent_state=recurrent_state,
                        has_previous_state=True,
                        expected_batch_size=1, expected_seq_len=1,
                        force_recurrent=True)
                    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
                    linear_conv_state[local_idx:local_idx+1] = next_conv.reshape(1, ane_dim1, ane_dim2)
                    linear_recurrent_state[local_idx:local_idx+1] = next_rec.to(linear_recurrent_state.dtype)
                    hidden_states = hidden_states + attn_out
                    post = layer.post_attention_layernorm(hidden_states)
                    hidden_states = hidden_states + layer.mlp(post)
                else:
                    # L-layer: use ON-THE-FLY RoPE instead of table lookup
                    x = layer.input_layernorm(hidden_states)
                    query_states, key_states, value_states, gate = layer.self_attn._project_qkvg(x)
                    query_states = layer.self_attn.q_norm(query_states)
                    key_states = layer.self_attn.k_norm(key_states)
                    # ON-THE-FLY RoPE: compute cos/sin without gather
                    cos, sin = self._rope_onthefly(position_ids, hidden_states.dtype, hidden_states.device)
                    from anemll.models.qwen3_5_model import apply_rotary_pos_emb_single
                    query_states, key_states = apply_rotary_pos_emb_single(
                        query_states, key_states, cos, sin, self._rotary_dim)
                    query_states = query_states.to(MODEL_DTYPE)
                    key_states = key_states.to(MODEL_DTYPE)
                    value_states = value_states.to(MODEL_DTYPE)
                    gate_states = gate.to(MODEL_DTYPE) if gate is not None else None
                    # KV cache update
                    pos = current_pos[0]
                    self.k_cache[local_idx, :, pos:pos+1, :] = key_states.squeeze(0)
                    self.v_cache[local_idx, :, pos:pos+1, :] = value_states.squeeze(0)
                    key_cache = self.k_cache[local_idx:local_idx+1].squeeze(0)
                    value_cache = self.v_cache[local_idx:local_idx+1].squeeze(0)
                    attn_out = layer.self_attn.forward_regular(
                        hidden_states=x, query_states=query_states,
                        kv_cache_layer=(key_cache, value_cache),
                        causal_mask=causal_mask, gate=gate_states)
                    hidden_states = hidden_states + attn_out
                    post = layer.post_attention_layernorm(hidden_states)
                    hidden_states = hidden_states + layer.mlp(post)

            if self.end_layer == len(self.model.model.layers):
                hidden_states = self.model.model.norm(hidden_states)
            return hidden_states, linear_conv_state, linear_recurrent_state

    return FFNWrapperRopeRewrite(model, start_layer, end_layer).eval()


def create_rewrite_b_model(model, cfg, chunk_idx):
    """Rewrite B: Replace softplus relu+abs with -log(sigmoid(-x)) formulation.

    Current code:
      sp = F.relu(x) + torch.log(1.0 + torch.exp(-torch.abs(x)))

    Rewrite:
      sp = -torch.log(torch.sigmoid(-x) + 1e-8)   [fp16 safe version]
    OR for fp32 math:
      sp = -torch.log(torch.sigmoid(-x))

    This eliminates F.relu (which may lower to greater_equal+select)
    and torch.abs (which may lower to greater_equal+select).
    """
    from anemll.models.qwen3_5_model import (
        Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
        Qwen35LinearLayoutStage
    )
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter, ane_conv_state_shape
    import types

    start_layer, end_layer = CHUNK_RANGES[chunk_idx]

    # Monkey-patch the layout stage's forward to use -log(sigmoid(-x))
    original_layout_forward = Qwen35LinearLayoutStage.forward

    def patched_layout_forward(self, conv_out_cf, z_cf, b_cf, a_cf, bsz, seq_len,
                                force_fp16_math=False):
        query_cf, key_cf, value_cf = torch.split(
            conv_out_cf, [self.key_dim, self.key_dim, self.value_dim], dim=1)
        query = self.from_channels_first_4d(query_cf).reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        key = self.from_channels_first_4d(key_cf).reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        value = self.from_channels_first_4d(value_cf).reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        z = self.from_channels_first_4d(z_cf).reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        b = self.from_channels_first_4d(b_cf)
        a = self.from_channels_first_4d(a_cf)
        beta = b.sigmoid()

        if force_fp16_math:
            x = a.to(MODEL_DTYPE) + self.dt_bias
            # ═══ REWRITE B: -log(sigmoid(-x)) instead of relu+abs ═══
            # sigmoid(-x) is always in (0, 1), numerically stable
            # For large x: sigmoid(-x) ≈ exp(-x) → -log(exp(-x)) = x ✓
            # For small x: sigmoid(-x) ≈ 1 → -log(1) = 0 ✓
            # Add eps for fp16 safety when x is very large
            sp = -torch.log(torch.sigmoid(-x) + 1e-7)
            g = -self.A_log.to(MODEL_DTYPE).exp() * sp
        else:
            x = a.float() + self.dt_bias
            # ═══ REWRITE B: -log(sigmoid(-x)) — fp32 path ═══
            # In fp32, sigmoid never reaches exactly 0
            sp = -torch.log(torch.sigmoid(-x))
            g = -self.A_log.float().exp() * sp

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)
        return query, key, value, g, beta, z

    # Apply monkey-patch to all F-layer layout stages
    for layer_idx in range(start_layer, end_layer):
        layer = model.model.layers[layer_idx]
        if layer.layer_type == "linear_attention":
            layer.self_attn.layout_stage.forward = types.MethodType(
                patched_layout_forward, layer.self_attn.layout_stage)

    return model  # Return the patched model (changes are in-place)


def restore_softplus(model, chunk_idx):
    """Restore original softplus after rewrite B testing."""
    from anemll.models.qwen3_5_model import Qwen35LinearLayoutStage
    start_layer, end_layer = CHUNK_RANGES[chunk_idx]
    for layer_idx in range(start_layer, end_layer):
        layer = model.model.layers[layer_idx]
        if layer.layer_type == "linear_attention":
            # Restore by re-binding to the class method
            del layer.self_attn.layout_stage.forward
            # This makes it fall back to the class method


def create_rewrite_c_model(model, cfg, chunk_idx):
    """Rewrite C: Pass cos/sin as model inputs (precomputed on host).

    Eliminates ALL internal RoPE computation from the model.
    Host precomputes cos[pos] and sin[pos] and passes them as inputs.
    """
    from anemll.models.qwen3_5_model import (
        Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
        apply_rotary_pos_emb_single
    )
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter, ane_conv_state_shape

    start_layer, end_layer = CHUNK_RANGES[chunk_idx]
    local_num_layers = end_layer - start_layer

    class FFNWrapperExternalRope(torch.nn.Module):
        """FFN wrapper that receives cos/sin as inputs instead of computing internally."""
        def __init__(self, model, start_layer, end_layer):
            super().__init__()
            self.model = model
            self.start_layer = start_layer
            self.end_layer = end_layer
            self.local_num_layers = end_layer - start_layer
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
                self._lin_rec_shape = (self.local_num_layers,
                                       cfg.text_config.linear_num_value_heads,
                                       cfg.text_config.linear_key_head_dim,
                                       cfg.text_config.linear_value_head_dim)
                self._has_linear = True
            else:
                self._has_linear = False
            self.states = Qwen35Converter.GetChunkLocalTransformerStates(
                model, self.local_num_layers, prefix="", split_full_attention_kv=True)
            # Get rotary dim
            for layer_idx in range(start_layer, end_layer):
                layer = model.model.layers[layer_idx]
                if layer.layer_type == "full_attention":
                    self._rotary_dim = layer.self_attn.rotary.rotary_dim
                    break

        def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                    linear_conv_state, linear_recurrent_state, rope_cos, rope_sin):
            for local_idx, layer_idx in enumerate(range(self.start_layer, self.end_layer)):
                layer = self.model.model.layers[layer_idx]
                if layer.layer_type == "linear_attention":
                    x = layer.input_layernorm(hidden_states)
                    conv_dim = layer.self_attn.conv_dim
                    conv_kernel = layer.self_attn.linear_conv_kernel_dim
                    conv_state_flat = linear_conv_state[local_idx:local_idx+1]
                    conv_state = conv_state_flat.reshape(1, conv_dim, conv_kernel)
                    recurrent_state = linear_recurrent_state[local_idx:local_idx+1]
                    attn_out, next_conv, next_rec = layer.self_attn.forward_regular(
                        hidden_states=x, conv_state=conv_state,
                        recurrent_state=recurrent_state, has_previous_state=True,
                        expected_batch_size=1, expected_seq_len=1, force_recurrent=True)
                    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
                    linear_conv_state[local_idx:local_idx+1] = next_conv.reshape(1, ane_dim1, ane_dim2)
                    linear_recurrent_state[local_idx:local_idx+1] = next_rec.to(linear_recurrent_state.dtype)
                    hidden_states = hidden_states + attn_out
                    post = layer.post_attention_layernorm(hidden_states)
                    hidden_states = hidden_states + layer.mlp(post)
                else:
                    x = layer.input_layernorm(hidden_states)
                    query_states, key_states, value_states, gate = layer.self_attn._project_qkvg(x)
                    query_states = layer.self_attn.q_norm(query_states)
                    key_states = layer.self_attn.k_norm(key_states)
                    # USE EXTERNAL cos/sin — NO internal gather!
                    query_states, key_states = apply_rotary_pos_emb_single(
                        query_states, key_states, rope_cos, rope_sin, self._rotary_dim)
                    query_states = query_states.to(MODEL_DTYPE)
                    key_states = key_states.to(MODEL_DTYPE)
                    value_states = value_states.to(MODEL_DTYPE)
                    gate_states = gate.to(MODEL_DTYPE) if gate is not None else None
                    pos = current_pos[0]
                    self.k_cache[local_idx, :, pos:pos+1, :] = key_states.squeeze(0)
                    self.v_cache[local_idx, :, pos:pos+1, :] = value_states.squeeze(0)
                    key_cache = self.k_cache[local_idx:local_idx+1].squeeze(0)
                    value_cache = self.v_cache[local_idx:local_idx+1].squeeze(0)
                    attn_out = layer.self_attn.forward_regular(
                        hidden_states=x, query_states=query_states,
                        kv_cache_layer=(key_cache, value_cache),
                        causal_mask=causal_mask, gate=gate_states)
                    hidden_states = hidden_states + attn_out
                    post = layer.post_attention_layernorm(hidden_states)
                    hidden_states = hidden_states + layer.mlp(post)

            if self.end_layer == len(self.model.model.layers):
                hidden_states = self.model.model.norm(hidden_states)
            return hidden_states, linear_conv_state, linear_recurrent_state

    return FFNWrapperExternalRope(model, start_layer, end_layer).eval()


def convert_rewrite(wrapper, cfg, rewrite_name, chunk_idx, extra_inputs=None):
    """Convert a rewrite wrapper to CoreML (with CoreML states for KV cache)."""
    from anemll.models.qwen3_5_model import MODEL_DTYPE, TEST_DEVICE
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter, ane_conv_state_shape

    hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, CTX), dtype=torch.float16, device=TEST_DEVICE)
    current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    local_num_layers = wrapper.local_num_layers
    if wrapper._has_linear:
        lin_conv = torch.zeros(wrapper._lin_conv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
        lin_rec = torch.zeros(wrapper._lin_rec_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    else:
        lin_conv = torch.zeros((local_num_layers, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        lin_rec = torch.zeros((local_num_layers, 1, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)

    inputs_list = [
        ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
        ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
        ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
        ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
    ]
    outputs_list = [
        ct.TensorType(name="output_hidden_states", dtype=np.float16),
        ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
        ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
    ]
    trace_args = (hidden_states, position_ids, causal_mask, current_pos, lin_conv, lin_rec)

    if extra_inputs:
        for inp_spec in extra_inputs:
            inputs_list.append(ct.TensorType(
                name=inp_spec['name'], shape=inp_spec['shape'], dtype=inp_spec['dtype']))
            trace_args = trace_args + (inp_spec['tensor'],)

    # Reset state buffers
    wrapper.k_cache.zero_()
    wrapper.v_cache.zero_()

    traced = torch.jit.trace(wrapper, trace_args, check_trace=False)
    wrapper.k_cache.zero_()
    wrapper.v_cache.zero_()
    traced.k_cache.zero_()
    traced.v_cache.zero_()

    mlmodel = ct.convert(
        traced,
        inputs=inputs_list,
        outputs=outputs_list,
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    out_path = os.path.join(ARTIFACT_DIR, f'{rewrite_name}_chunk{chunk_idx}_decode.mlpackage')
    mlmodel.save(out_path)
    return out_path


def convert_rewrite_stateless(wrapper, cfg, rewrite_name, chunk_idx):
    """Convert a stateless rewrite wrapper (no CoreML states — KV cache as I/O)."""
    from anemll.models.qwen3_5_model import MODEL_DTYPE, TEST_DEVICE
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter, ane_conv_state_shape

    hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, CTX), dtype=torch.float16, device=TEST_DEVICE)
    current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    local_num_layers = wrapper.local_num_layers
    if wrapper._has_linear:
        lin_conv = torch.zeros(wrapper._lin_conv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
        lin_rec = torch.zeros(wrapper._lin_rec_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    else:
        lin_conv = torch.zeros((local_num_layers, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        lin_rec = torch.zeros((local_num_layers, 1, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)

    # KV cache as regular I/O tensors
    k_cache = torch.zeros(wrapper._kv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    v_cache = torch.zeros(wrapper._kv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)

    trace_args = (hidden_states, position_ids, causal_mask, current_pos,
                  lin_conv, lin_rec, k_cache, v_cache)

    traced = torch.jit.trace(wrapper, trace_args, check_trace=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
            ct.TensorType(name="k_cache_in", shape=k_cache.shape, dtype=np.float16),
            ct.TensorType(name="v_cache_in", shape=v_cache.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
            ct.TensorType(name="k_cache_out", dtype=np.float16),
            ct.TensorType(name="v_cache_out", dtype=np.float16),
        ],
        # NO states= parameter! KV cache is regular I/O
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    out_path = os.path.join(ARTIFACT_DIR, f'{rewrite_name}_chunk{chunk_idx}_decode.mlpackage')
    mlmodel.save(out_path)
    return out_path


def create_rewrite_d_model(model, cfg, chunk_idx):
    """Rewrite D: On-the-fly RoPE + Stateless KV cache (I/O tensors, no CoreML state).

    Eliminates ALL hostile ops:
    - gather/greater_equal/select from RoPE (via on-the-fly computation)
    - read_state/write_state from KV cache (via regular I/O tensors)
    """
    from anemll.models.qwen3_5_model import (
        Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
        apply_rotary_pos_emb_single
    )
    from anemll.ane_converter.qwen3_5_converter import ane_conv_state_shape

    start_layer, end_layer = CHUNK_RANGES[chunk_idx]
    local_num_layers = end_layer - start_layer

    class FFNWrapperStateless(torch.nn.Module):
        """FFN wrapper with: 1) on-the-fly RoPE, 2) KV cache as regular I/O."""
        def __init__(self, model, start_layer, end_layer):
            super().__init__()
            self.model = model
            self.start_layer = start_layer
            self.end_layer = end_layer
            self.local_num_layers = end_layer - start_layer
            cfg = model.config
            # KV cache shape for I/O (NOT register_buffer, NOT CoreML state)
            self._kv_shape = (
                self.local_num_layers,
                cfg.num_key_value_heads,
                cfg.state_length,
                cfg.head_dim,
            )
            if cfg.has_linear_attention():
                conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                           + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
                conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
                ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
                self._lin_conv_shape = (self.local_num_layers, ane_dim1, ane_dim2)
                self._lin_rec_shape = (self.local_num_layers,
                                       cfg.text_config.linear_num_value_heads,
                                       cfg.text_config.linear_key_head_dim,
                                       cfg.text_config.linear_value_head_dim)
                self._has_linear = True
            else:
                self._has_linear = False

            # RoPE inv_freq for on-the-fly computation
            for layer_idx in range(start_layer, end_layer):
                layer = model.model.layers[layer_idx]
                if layer.layer_type == "full_attention":
                    self.register_buffer("rope_inv_freq",
                                        layer.self_attn.rotary.inv_freq.clone())
                    self._rotary_dim = layer.self_attn.rotary.rotary_dim
                    break

        def _rope_onthefly(self, position_ids, dtype, device):
            if position_ids.dim() > 1:
                pos_ids = position_ids.squeeze(0)
            else:
                pos_ids = position_ids
            t = pos_ids.to(dtype=torch.float32).unsqueeze(-1)
            freqs = t * self.rope_inv_freq.unsqueeze(0)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().unsqueeze(0).to(dtype)
            sin = emb.sin().unsqueeze(0).to(dtype)
            return cos, sin

        def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                    linear_conv_state, linear_recurrent_state, k_cache, v_cache):
            for local_idx, layer_idx in enumerate(range(self.start_layer, self.end_layer)):
                layer = self.model.model.layers[layer_idx]
                if layer.layer_type == "linear_attention":
                    x = layer.input_layernorm(hidden_states)
                    conv_dim = layer.self_attn.conv_dim
                    conv_kernel = layer.self_attn.linear_conv_kernel_dim
                    conv_state_flat = linear_conv_state[local_idx:local_idx+1]
                    conv_state = conv_state_flat.reshape(1, conv_dim, conv_kernel)
                    recurrent_state = linear_recurrent_state[local_idx:local_idx+1]
                    attn_out, next_conv, next_rec = layer.self_attn.forward_regular(
                        hidden_states=x, conv_state=conv_state,
                        recurrent_state=recurrent_state, has_previous_state=True,
                        expected_batch_size=1, expected_seq_len=1, force_recurrent=True)
                    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
                    linear_conv_state[local_idx:local_idx+1] = next_conv.reshape(1, ane_dim1, ane_dim2)
                    linear_recurrent_state[local_idx:local_idx+1] = next_rec.to(linear_recurrent_state.dtype)
                    hidden_states = hidden_states + attn_out
                    post = layer.post_attention_layernorm(hidden_states)
                    hidden_states = hidden_states + layer.mlp(post)
                else:
                    x = layer.input_layernorm(hidden_states)
                    query_states, key_states, value_states, gate = layer.self_attn._project_qkvg(x)
                    query_states = layer.self_attn.q_norm(query_states)
                    key_states = layer.self_attn.k_norm(key_states)
                    cos, sin = self._rope_onthefly(position_ids, hidden_states.dtype, hidden_states.device)
                    query_states, key_states = apply_rotary_pos_emb_single(
                        query_states, key_states, cos, sin, self._rotary_dim)
                    query_states = query_states.to(MODEL_DTYPE)
                    key_states = key_states.to(MODEL_DTYPE)
                    value_states = value_states.to(MODEL_DTYPE)
                    gate_states = gate.to(MODEL_DTYPE) if gate is not None else None
                    # STATELESS: KV cache as regular tensor I/O
                    pos = current_pos[0]
                    k_cache[local_idx, :, pos:pos+1, :] = key_states.squeeze(0)
                    v_cache[local_idx, :, pos:pos+1, :] = value_states.squeeze(0)
                    key_cache = k_cache[local_idx:local_idx+1].squeeze(0)
                    value_cache = v_cache[local_idx:local_idx+1].squeeze(0)
                    attn_out = layer.self_attn.forward_regular(
                        hidden_states=x, query_states=query_states,
                        kv_cache_layer=(key_cache, value_cache),
                        causal_mask=causal_mask, gate=gate_states)
                    hidden_states = hidden_states + attn_out
                    post = layer.post_attention_layernorm(hidden_states)
                    hidden_states = hidden_states + layer.mlp(post)

            if self.end_layer == len(self.model.model.layers):
                hidden_states = self.model.model.norm(hidden_states)
            return hidden_states, linear_conv_state, linear_recurrent_state, k_cache, v_cache

    return FFNWrapperStateless(model, start_layer, end_layer).eval()


def create_rewrite_e_split_models(model, cfg, chunk_idx):
    """Rewrite E: Split FLLL chunk into separate F model (no KV cache)
    and LLL model (KV cache only, like chunk 0).

    This isolates the F-layer so it doesn't fragment the L-layers' KV cache
    graph, allowing the LLL sub-model to achieve chunk-0-like ANE placement.
    """
    from anemll.models.qwen3_5_model import (
        Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
        apply_rotary_pos_emb_single
    )
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter, ane_conv_state_shape

    start_layer, end_layer = CHUNK_RANGES[chunk_idx]

    # Find F-layers and L-layers
    f_layers = []
    l_layers = []
    for layer_idx in range(start_layer, end_layer):
        layer = model.model.layers[layer_idx]
        if layer.layer_type == "linear_attention":
            f_layers.append(layer_idx)
        else:
            l_layers.append(layer_idx)

    if not f_layers:
        return None, None

    # ── F-only wrapper: processes linear attention layers, no KV cache ──
    class FOnlyWrapper(torch.nn.Module):
        def __init__(self, model, f_layers):
            super().__init__()
            self.model = model
            self.f_layers = f_layers
            self.local_num_layers = len(f_layers)
            cfg = model.config
            conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                       + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
            conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
            ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
            self._lin_conv_shape = (self.local_num_layers, ane_dim1, ane_dim2)
            self._lin_rec_shape = (self.local_num_layers,
                                   cfg.text_config.linear_num_value_heads,
                                   cfg.text_config.linear_key_head_dim,
                                   cfg.text_config.linear_value_head_dim)
            self._has_linear = True

        def forward(self, hidden_states, linear_conv_state, linear_recurrent_state):
            for local_idx, layer_idx in enumerate(self.f_layers):
                layer = self.model.model.layers[layer_idx]
                x = layer.input_layernorm(hidden_states)
                conv_dim = layer.self_attn.conv_dim
                conv_kernel = layer.self_attn.linear_conv_kernel_dim
                conv_state_flat = linear_conv_state[local_idx:local_idx+1]
                conv_state = conv_state_flat.reshape(1, conv_dim, conv_kernel)
                recurrent_state = linear_recurrent_state[local_idx:local_idx+1]
                attn_out, next_conv, next_rec = layer.self_attn.forward_regular(
                    hidden_states=x, conv_state=conv_state,
                    recurrent_state=recurrent_state, has_previous_state=True,
                    expected_batch_size=1, expected_seq_len=1, force_recurrent=True)
                ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
                linear_conv_state[local_idx:local_idx+1] = next_conv.reshape(1, ane_dim1, ane_dim2)
                linear_recurrent_state[local_idx:local_idx+1] = next_rec.to(linear_recurrent_state.dtype)
                hidden_states = hidden_states + attn_out
                post = layer.post_attention_layernorm(hidden_states)
                hidden_states = hidden_states + layer.mlp(post)
            return hidden_states, linear_conv_state, linear_recurrent_state

    # ── LLL-only wrapper: processes full-attention layers with on-the-fly RoPE ──
    class LLLOnlyWrapper(torch.nn.Module):
        def __init__(self, model, l_layers):
            super().__init__()
            self.model = model
            self.l_layers = l_layers
            self.local_num_layers = len(l_layers)
            cfg = model.config
            self.register_buffer("k_cache", torch.zeros(
                (self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE, device=TEST_DEVICE))
            self.register_buffer("v_cache", torch.zeros(
                (self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE, device=TEST_DEVICE))
            self._has_linear = False
            self.states = Qwen35Converter.GetChunkLocalTransformerStates(
                model, self.local_num_layers, prefix="", split_full_attention_kv=True)
            # On-the-fly RoPE
            layer = model.model.layers[l_layers[0]]
            self.register_buffer("rope_inv_freq", layer.self_attn.rotary.inv_freq.clone())
            self._rotary_dim = layer.self_attn.rotary.rotary_dim

        def _rope_onthefly(self, position_ids, dtype, device):
            if position_ids.dim() > 1:
                pos_ids = position_ids.squeeze(0)
            else:
                pos_ids = position_ids
            t = pos_ids.to(dtype=torch.float32).unsqueeze(-1)
            freqs = t * self.rope_inv_freq.unsqueeze(0)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().unsqueeze(0).to(dtype)
            sin = emb.sin().unsqueeze(0).to(dtype)
            return cos, sin

        def forward(self, hidden_states, position_ids, causal_mask, current_pos):
            cos, sin = self._rope_onthefly(position_ids, hidden_states.dtype, hidden_states.device)
            for local_idx, layer_idx in enumerate(self.l_layers):
                layer = self.model.model.layers[layer_idx]
                x = layer.input_layernorm(hidden_states)
                query_states, key_states, value_states, gate = layer.self_attn._project_qkvg(x)
                query_states = layer.self_attn.q_norm(query_states)
                key_states = layer.self_attn.k_norm(key_states)
                query_states, key_states = apply_rotary_pos_emb_single(
                    query_states, key_states, cos, sin, self._rotary_dim)
                query_states = query_states.to(MODEL_DTYPE)
                key_states = key_states.to(MODEL_DTYPE)
                value_states = value_states.to(MODEL_DTYPE)
                gate_states = gate.to(MODEL_DTYPE) if gate is not None else None
                pos = current_pos[0]
                self.k_cache[local_idx, :, pos:pos+1, :] = key_states.squeeze(0)
                self.v_cache[local_idx, :, pos:pos+1, :] = value_states.squeeze(0)
                key_cache = self.k_cache[local_idx:local_idx+1].squeeze(0)
                value_cache = self.v_cache[local_idx:local_idx+1].squeeze(0)
                attn_out = layer.self_attn.forward_regular(
                    hidden_states=x, query_states=query_states,
                    kv_cache_layer=(key_cache, value_cache),
                    causal_mask=causal_mask, gate=gate_states)
                hidden_states = hidden_states + attn_out
                post = layer.post_attention_layernorm(hidden_states)
                hidden_states = hidden_states + layer.mlp(post)
            if end_layer == len(model.model.layers):
                hidden_states = model.model.norm(hidden_states)
            return hidden_states

    f_wrapper = FOnlyWrapper(model, f_layers).eval()
    l_wrapper = LLLOnlyWrapper(model, l_layers).eval()
    return f_wrapper, l_wrapper


def convert_rewrite_e_f(wrapper, cfg, chunk_idx):
    """Convert F-only sub-model (no states, no KV cache)."""
    from anemll.models.qwen3_5_model import MODEL_DTYPE, TEST_DEVICE

    hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    lin_conv = torch.zeros(wrapper._lin_conv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    lin_rec = torch.zeros(wrapper._lin_rec_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)

    traced = torch.jit.trace(wrapper, (hidden_states, lin_conv, lin_rec), check_trace=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    out_path = os.path.join(ARTIFACT_DIR, f'rewrite_e_f_only_chunk{chunk_idx}_decode.mlpackage')
    mlmodel.save(out_path)
    return out_path


def convert_rewrite_e_l(wrapper, cfg, chunk_idx):
    """Convert LLL-only sub-model (with states for KV cache, on-the-fly RoPE)."""
    from anemll.models.qwen3_5_model import MODEL_DTYPE, TEST_DEVICE

    hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, CTX), dtype=torch.float16, device=TEST_DEVICE)
    current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

    wrapper.k_cache.zero_()
    wrapper.v_cache.zero_()
    traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos),
                              check_trace=False)
    wrapper.k_cache.zero_()
    wrapper.v_cache.zero_()
    traced.k_cache.zero_()
    traced.v_cache.zero_()

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    out_path = os.path.join(ARTIFACT_DIR, f'rewrite_e_lll_only_chunk{chunk_idx}_decode.mlpackage')
    mlmodel.save(out_path)
    return out_path


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3-4: Export rewrites + Measure placement
# ═══════════════════════════════════════════════════════════════════════

def measure_placement(model_path, fn_name=None, pos=50, warmup=3, trials=10):
    """Measure ANE vs CPU utilization for a model."""
    import resource
    model_ne = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE,
                                  function_name=fn_name)
    model_cpu = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY,
                                   function_name=fn_name)

    # Build input — only use ACTUAL inputs, not states
    spec = ct.utils.load_spec(model_path)
    if fn_name and fn_name in spec.mlProgram.functions:
        fn = spec.mlProgram.functions[fn_name]
    else:
        fn = list(spec.mlProgram.functions.values())[0]

    # Get set of state names to exclude from inputs
    state_names = set()
    try:
        for state_desc in fn.states:
            state_names.add(state_desc.name)
    except Exception:
        # Fallback: exclude common state names
        state_names = {'k_cache', 'v_cache', 'kv_cache_0'}

    inp = {}
    for desc in fn.inputs:
        name = desc.name
        if name in state_names:
            continue  # Skip states — they're handled via make_state()
        try:
            shape = [d.constant.size for d in desc.type.tensorType.dimensions]
            dt = desc.type.tensorType.dataType
            np_dt = {1: np.float32, 10: np.float16, 11: np.float16, 5: np.int32, 7: np.int64}.get(dt, np.float16)
        except Exception:
            continue
        if name == 'causal_mask':
            data = np.full(shape, -65504.0, dtype=np_dt)
            data[..., :pos+1] = 0
        elif name == 'current_pos':
            data = np.array([pos], dtype=np_dt)
        elif name == 'position_ids':
            data = np.array([pos], dtype=np_dt)
        else:
            data = np.zeros(shape, dtype=np_dt)
        inp[name] = data

    # Initialize states (may not exist for stateless models)
    try:
        state_ne = model_ne.make_state()
        state_cpu = model_cpu.make_state()
    except Exception:
        state_ne = None
        state_cpu = None

    # Warmup
    for _ in range(warmup):
        if state_ne is not None:
            model_ne.predict(inp, state=state_ne)
        else:
            model_ne.predict(inp)
        if state_cpu is not None:
            model_cpu.predict(inp, state=state_cpu)
        else:
            model_cpu.predict(inp)

    # Measure ANE path
    times_ne = []
    for _ in range(trials):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0_cpu = time.process_time()
        t0_wall = time.perf_counter()
        if state_ne is not None:
            model_ne.predict(inp, state=state_ne)
        else:
            model_ne.predict(inp)
        wall = time.perf_counter() - t0_wall
        cpu_t = time.process_time() - t0_cpu
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        vctx = r1.ru_nvcsw - r0.ru_nvcsw
        times_ne.append({'wall': wall * 1000, 'cpu': cpu_t * 1000, 'vctx': vctx})

    # Measure CPU path
    times_cpu = []
    for _ in range(trials):
        t0_cpu = time.process_time()
        t0_wall = time.perf_counter()
        if state_cpu is not None:
            model_cpu.predict(inp, state=state_cpu)
        else:
            model_cpu.predict(inp)
        wall = time.perf_counter() - t0_wall
        cpu_t = time.process_time() - t0_cpu
        times_cpu.append({'wall': wall * 1000, 'cpu': cpu_t * 1000})

    # Compute medians
    ne_wall = sorted([t['wall'] for t in times_ne])[len(times_ne)//2]
    ne_cpu = sorted([t['cpu'] for t in times_ne])[len(times_ne)//2]
    ne_vctx = sorted([t['vctx'] for t in times_ne])[len(times_ne)//2]
    cpu_wall = sorted([t['wall'] for t in times_cpu])[len(times_cpu)//2]
    cpu_cpu = sorted([t['cpu'] for t in times_cpu])[len(times_cpu)//2]

    cpu_frac = ne_cpu / ne_wall if ne_wall > 0 else 0
    ane_frac = max(0, 1.0 - cpu_frac)
    speedup = cpu_wall / ne_wall if ne_wall > 0 else 0

    return {
        'ne_wall': ne_wall, 'ne_cpu': ne_cpu, 'ne_vctx': ne_vctx,
        'cpu_wall': cpu_wall, 'cpu_cpu': cpu_cpu,
        'cpu_frac': cpu_frac, 'ane_frac': ane_frac,
        'speedup': speedup,
    }


def run_phase3_4(model, cfg, chunk_idx=1):
    """Phase 3-4: Export rewrites and measure placement."""
    from anemll.models.qwen3_5_model import MODEL_DTYPE, TEST_DEVICE
    from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

    print("=" * 70)
    print("  PHASE 3: Implement and Export Rewrites")
    print("=" * 70)

    start_layer, end_layer = CHUNK_RANGES[chunk_idx]
    results = {}

    # ── Baseline (standard conversion) ──
    print(f"\n  ── BASELINE: Standard chunk {chunk_idx} decode ──")
    baseline_path = os.path.join(ARTIFACT_DIR, f'baseline_chunk{chunk_idx}_decode.mlpackage')
    if not os.path.exists(baseline_path):
        print("  Converting baseline...")
        t0 = time.time()
        conv = Qwen35Converter(
            model, context_length=CTX, batch_size=BATCH_SIZE,
            num_chunks=NUM_CHUNKS, lut_bits=4, per_channel=4,
            compute_precision="float32")
        ml = conv.convert_part_2(
            model, chunk_idx=chunk_idx, total_chunks=NUM_CHUNKS,
            override_start_layer=start_layer, override_end_layer=end_layer)
        ml.save(baseline_path)
        del ml; gc.collect()
        print(f"  Saved in {time.time()-t0:.1f}s")
    else:
        print(f"  Using existing baseline: {baseline_path}")

    # ── Rewrite A: On-the-fly RoPE ──
    print(f"\n  ── REWRITE A: On-the-fly RoPE computation ──")
    rw_a_path = os.path.join(ARTIFACT_DIR, f'rewrite_a_rope_chunk{chunk_idx}_decode.mlpackage')
    if not os.path.exists(rw_a_path):
        print("  Building rewrite A wrapper...")
        wrapper_a = create_rewrite_a_model(model, cfg, chunk_idx)
        print("  Converting rewrite A...")
        t0 = time.time()
        rw_a_path = convert_rewrite(wrapper_a, cfg, 'rewrite_a_rope', chunk_idx)
        del wrapper_a; gc.collect()
        print(f"  Saved in {time.time()-t0:.1f}s")
    else:
        print(f"  Using existing: {rw_a_path}")

    # ── Rewrite B: Softplus -log(sigmoid(-x)) ──
    # ANALYSIS: Phase 1 showed greater_equal/select come from RoPE bounds
    # clamping, NOT from softplus. Skipping rewrite B.
    rw_b_path = None
    print(f"\n  ── REWRITE B: SKIPPED (hostile ops are RoPE, not softplus) ──")

    # ── Rewrite A+B not needed since B is ineffective ──
    rw_ab_path = None
    print(f"\n  ── REWRITE A+B: SKIPPED ──")

    # ── Rewrite D: On-the-fly RoPE + Stateless KV cache (I/O, no CoreML state) ──
    print(f"\n  ── REWRITE D: On-the-fly RoPE + Stateless KV cache ──")
    rw_d_path = os.path.join(ARTIFACT_DIR, f'rewrite_d_stateless_chunk{chunk_idx}_decode.mlpackage')
    if not os.path.exists(rw_d_path):
        print("  Building rewrite D wrapper...")
        wrapper_d = create_rewrite_d_model(model, cfg, chunk_idx)
        print("  Converting rewrite D (no CoreML states)...")
        t0 = time.time()
        rw_d_path = convert_rewrite_stateless(wrapper_d, cfg, 'rewrite_d_stateless', chunk_idx)
        del wrapper_d; gc.collect()
        print(f"  Saved in {time.time()-t0:.1f}s")
    else:
        print(f"  Using existing: {rw_d_path}")

    # ── Phase 4: MIL Analysis + Placement Measurement ──
    print("\n" + "=" * 70)
    print("  PHASE 4: MIL Analysis + Placement Measurement")
    print("=" * 70)

    variants = [
        ('BASELINE', baseline_path),
        ('REWRITE_A (on-the-fly RoPE)', rw_a_path),
    ]
    if rw_b_path:
        variants.append(('REWRITE_B (softplus)', rw_b_path))
    if rw_ab_path:
        variants.append(('REWRITE_A+B (both)', rw_ab_path))
    if rw_d_path:
        variants.append(('REWRITE_D (stateless+rope)', rw_d_path))

    # ── Rewrite E: Split FLLL into F + LLL sub-models ──
    print(f"\n  ── REWRITE E: Split FLLL → separate F model + LLL model ──")
    rw_e_f_path = os.path.join(ARTIFACT_DIR, f'rewrite_e_f_only_chunk{chunk_idx}_decode.mlpackage')
    rw_e_l_path = os.path.join(ARTIFACT_DIR, f'rewrite_e_lll_only_chunk{chunk_idx}_decode.mlpackage')
    if not os.path.exists(rw_e_f_path) or not os.path.exists(rw_e_l_path):
        print("  Building split F/LLL wrappers...")
        f_wrapper, l_wrapper = create_rewrite_e_split_models(model, cfg, chunk_idx)
        if f_wrapper is not None:
            t0 = time.time()
            print("  Converting F-only sub-model...")
            rw_e_f_path = convert_rewrite_e_f(f_wrapper, cfg, chunk_idx)
            del f_wrapper; gc.collect()
            print(f"    Saved F in {time.time()-t0:.1f}s")
            t0 = time.time()
            print("  Converting LLL-only sub-model (with on-the-fly RoPE)...")
            rw_e_l_path = convert_rewrite_e_l(l_wrapper, cfg, chunk_idx)
            del l_wrapper; gc.collect()
            print(f"    Saved LLL in {time.time()-t0:.1f}s")
        else:
            print("  SKIP: chunk has no F-layer to split")
            rw_e_f_path = None
            rw_e_l_path = None
    else:
        print(f"  Using existing: F={rw_e_f_path}, LLL={rw_e_l_path}")

    if rw_e_f_path:
        variants.append(('REWRITE_E: F-only (no KV cache)', rw_e_f_path))
    if rw_e_l_path:
        variants.append(('REWRITE_E: LLL-only (like chunk0)', rw_e_l_path))

    for name, path in variants:
        print(f"\n  ── {name} ──")
        if not os.path.exists(path):
            print(f"    SKIP: {path} not found")
            continue

        # MIL analysis
        spec = ct.utils.load_spec(path)
        fns = list(spec.mlProgram.functions.keys())
        fn_name = fns[0] if fns else None
        print(f"    Functions: {fns}")

        hostile_ops, all_ops = deep_mil_diagnosis(path, fn_name)
        type_counts = Counter(h['type'] for h in hostile_ops)
        rt_ops = [o for o in all_ops if o['type'] not in ('const', 'constexpr_lut_to_dense',
                                                           'constexpr_affine_dequantize')]
        print(f"    Total ops: {len(all_ops)}, Runtime: {len(rt_ops)}, Hostile: {len(hostile_ops)}")
        if hostile_ops:
            print(f"    Hostile breakdown: {dict(type_counts)}")
            for h in hostile_ops:
                print(f"      [{h['idx']:4d}] {h['type']:25s} → {h['out_name'][:60]}")

        # Placement measurement
        print(f"\n    Measuring ANE/CPU placement...")
        try:
            metrics = measure_placement(path, fn_name=fn_name)
            print(f"    CPU_AND_NE wall: {metrics['ne_wall']:.2f}ms")
            print(f"    CPU_ONLY   wall: {metrics['cpu_wall']:.2f}ms")
            print(f"    CPU fraction:    {metrics['cpu_frac']:.0%}")
            print(f"    ANE fraction:    {metrics['ane_frac']:.0%}")
            print(f"    Speedup (CPU/NE): {metrics['speedup']:.2f}x")
            print(f"    Vol ctx switches: {metrics['ne_vctx']}")
            results[name] = {**metrics, 'hostile_count': len(hostile_ops),
                            'hostile_breakdown': dict(type_counts),
                            'runtime_ops': len(rt_ops)}
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}

    return results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: Correctness Validation
# ═══════════════════════════════════════════════════════════════════════

def run_phase5(model, cfg, chunk_idx=1):
    """Phase 5: Validate rewrite correctness by comparing outputs."""
    print("\n" + "=" * 70)
    print("  PHASE 5: Correctness Validation")
    print("=" * 70)

    baseline_path = os.path.join(ARTIFACT_DIR, f'baseline_chunk{chunk_idx}_decode.mlpackage')
    rewrite_paths = {
        'REWRITE_A': os.path.join(ARTIFACT_DIR, f'rewrite_a_rope_chunk{chunk_idx}_decode.mlpackage'),
    }

    # Load baseline
    spec = ct.utils.load_spec(baseline_path)
    fns = list(spec.mlProgram.functions.keys())
    fn_name = fns[0]

    baseline_model = ct.models.MLModel(baseline_path, compute_units=ct.ComputeUnit.CPU_ONLY,
                                        function_name=fn_name)
    baseline_state = baseline_model.make_state()

    # Build test inputs at different positions
    test_positions = [0, 10, 50, 100, 500]
    hidden_dim = cfg.hidden_size

    for rw_name, rw_path in rewrite_paths.items():
        if not os.path.exists(rw_path):
            print(f"\n  {rw_name}: SKIP (not found)")
            continue

        rw_spec = ct.utils.load_spec(rw_path)
        rw_fns = list(rw_spec.mlProgram.functions.keys())
        rw_fn = rw_fns[0]
        rw_model = ct.models.MLModel(rw_path, compute_units=ct.ComputeUnit.CPU_ONLY,
                                      function_name=rw_fn)
        rw_state = rw_model.make_state()

        print(f"\n  {rw_name} vs BASELINE:")
        max_abs_diffs = []
        cos_sims = []

        for pos in test_positions:
            # Build inputs
            hs = np.random.randn(1, 1, hidden_dim).astype(np.float16) * 0.01
            mask = np.full((1, 1, 1, CTX), -65504, dtype=np.float16)
            mask[..., :pos+1] = 0

            base_inp = {
                'hidden_states': hs,
                'position_ids': np.array([pos], dtype=np.int32),
                'causal_mask': mask,
                'current_pos': np.array([pos], dtype=np.int32),
            }
            # Add lin states if present, exclude CoreML state names
            state_names_check = {'k_cache', 'v_cache', 'kv_cache_0'}
            for desc in spec.mlProgram.functions[fn_name].inputs:
                if desc.name not in base_inp and desc.name not in state_names_check:
                    shape = [d.constant.size for d in desc.type.tensorType.dimensions]
                    dt = desc.type.tensorType.dataType
                    np_dt = {1: np.float32, 10: np.float16, 11: np.float16, 5: np.int32}.get(dt, np.float16)
                    base_inp[desc.name] = np.zeros(shape, dtype=np_dt)

            rw_inp = dict(base_inp)
            # Handle extra inputs for rewrite C (rope_cos, rope_sin)
            for desc in rw_spec.mlProgram.functions[rw_fn].inputs:
                if desc.name not in rw_inp and desc.name not in state_names_check:
                    shape = [d.constant.size for d in desc.type.tensorType.dimensions]
                    dt = desc.type.tensorType.dataType
                    np_dt = {1: np.float32, 10: np.float16, 11: np.float16, 5: np.int32}.get(dt, np.float16)
                    rw_inp[desc.name] = np.zeros(shape, dtype=np_dt)

            base_out = baseline_model.predict(base_inp, state=baseline_state)
            rw_out = rw_model.predict(rw_inp, state=rw_state)

            # Compare output_hidden_states
            b = base_out['output_hidden_states'].flatten().astype(np.float32)
            r = rw_out['output_hidden_states'].flatten().astype(np.float32)
            max_abs = np.max(np.abs(b - r))
            cos_sim = np.dot(b, r) / (np.linalg.norm(b) * np.linalg.norm(r) + 1e-10)
            max_abs_diffs.append(max_abs)
            cos_sims.append(cos_sim)
            print(f"    pos={pos:4d}: max_abs_diff={max_abs:.6f}  cos_sim={cos_sim:.6f}")

        avg_cos = np.mean(cos_sims)
        max_diff = max(max_abs_diffs)
        print(f"    SUMMARY: avg_cos={avg_cos:.6f}  max_diff={max_diff:.6f}")
        if avg_cos > 0.99:
            print(f"    → PASS: Very high parity")
        elif avg_cos > 0.95:
            print(f"    → ACCEPTABLE: Minor numerical drift")
        else:
            print(f"    → WARN: Significant divergence")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6: Summary Report
# ═══════════════════════════════════════════════════════════════════════

def run_phase6(results):
    """Generate final comparison report."""
    print("\n" + "=" * 70)
    print("  PHASE 6: Final Comparison Report")
    print("=" * 70)

    if not results:
        print("  No results to report")
        return

    print(f"\n  {'Variant':<35s} {'Wall(ms)':>8s} {'CPU%':>6s} {'ANE%':>6s} {'Speed':>6s} {'Hostile':>8s}")
    print(f"  {'─'*35} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")

    for name, m in results.items():
        if 'error' in m:
            print(f"  {name:<35s} ERROR: {m['error']}")
            continue
        print(f"  {name:<35s} {m['ne_wall']:>7.2f} {m['cpu_frac']:>5.0%} {m['ane_frac']:>5.0%}"
              f" {m['speedup']:>5.2f}x {m['hostile_count']:>7d}")
        if m.get('hostile_breakdown'):
            print(f"  {'':35s} hostile: {m['hostile_breakdown']}")

    # Recommendations
    print(f"\n  ── Recommendations ──")
    baseline = results.get('BASELINE', {})
    for name, m in results.items():
        if name == 'BASELINE' or 'error' in m:
            continue
        b_hostile = baseline.get('hostile_count', 0)
        m_hostile = m.get('hostile_count', 0)
        b_ane = baseline.get('ane_frac', 0)
        m_ane = m.get('ane_frac', 0)
        hostile_delta = m_hostile - b_hostile
        ane_delta = m_ane - b_ane
        wall_delta = m['ne_wall'] - baseline.get('ne_wall', m['ne_wall'])
        print(f"\n  {name}:")
        print(f"    Hostile ops: {b_hostile} → {m_hostile} ({hostile_delta:+d})")
        print(f"    ANE fraction: {b_ane:.0%} → {m_ane:.0%} ({ane_delta:+.0%})")
        print(f"    Wall time: {baseline.get('ne_wall', 0):.2f}ms → {m['ne_wall']:.2f}ms ({wall_delta:+.2f}ms)")
        if m_hostile == 0 and m_ane > b_ane:
            print(f"    ★ RECOMMENDED: All hostile ops eliminated, ANE improved")
        elif m_hostile < b_hostile and m_ane > b_ane:
            print(f"    ★ PROMISING: Reduced hostile ops, ANE improved")
        elif m_hostile < b_hostile:
            print(f"    → PARTIAL: Reduced hostile ops but ANE unchanged")
        else:
            print(f"    → NO IMPROVEMENT: Hostile ops unchanged")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ANE Rewrite Investigation')
    parser.add_argument('--phase', type=int, default=0, help='Run specific phase (0=all)')
    parser.add_argument('--chunk', type=int, default=1, help='Chunk index to investigate')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing artifacts')
    args = parser.parse_args()

    print("═" * 70)
    print("  ANE REWRITE INVESTIGATION")
    print(f"  Chunk: {args.chunk}  Layers: {CHUNK_RANGES[args.chunk]}")
    print(f"  CTX={CTX}  BATCH={BATCH_SIZE}  CHUNKS={NUM_CHUNKS}")
    print("═" * 70)

    if args.phase in (0, 1):
        phase1_results = run_phase1(args.chunk)

    model = cfg = None
    if args.phase in (0, 2):
        model, cfg, hostile_ops, all_ops = run_phase2()

    if args.phase in (0, 3, 4):
        if model is None:
            from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
            print("\nLoading model weights...")
            cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
            cfg.context_length = CTX
            cfg.state_length = CTX
            model = Qwen35ForCausalLM(cfg)
            assert model.load_pretrained_weights(HF_MODEL)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        results = run_phase3_4(model, cfg, args.chunk)

    if args.phase in (0, 5):
        if model is None:
            from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config
            print("\nLoading model weights...")
            cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
            cfg.context_length = CTX
            cfg.state_length = CTX
            model = Qwen35ForCausalLM(cfg)
            assert model.load_pretrained_weights(HF_MODEL)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
        run_phase5(model, cfg, args.chunk)

    if args.phase in (0, 6):
        if 'results' in dir():
            run_phase6(results)

    print("\n" + "═" * 70)
    print("  INVESTIGATION COMPLETE")
    print("═" * 70)


if __name__ == '__main__':
    main()
