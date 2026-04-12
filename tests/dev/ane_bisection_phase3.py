#!/usr/bin/env python3
"""
ANE Bisection Phase 3: F + L layer combinations.

Phase 1-2 found: ALL F-layer combinations achieve 88-95% ANE.
Actual chunks have FLLL pattern (1 F + 3 L). Something in L layers may kill ANE.

This script tests with actual Qwen3.5 model classes:
  1F:     Single F layer (layer 3)               — baseline: ~92% ANE
  1F+1L:  F + 1 L layer (layers 3-4)
  1F+2L:  F + 2 L layers (layers 3-5)
  1F+3L:  F + 3 L layers (layers 3-6)            — matches actual chunk pattern
  3L:     3 L layers only (layers 0-2)            — chunk 0 pattern (L-only)
  1L:     Single L layer (layer 0)
"""

import argparse
import gc
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

import coremltools as ct

REPO_ROOT = "/Volumes/MySSD/Anemll"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
os.chdir(REPO_ROOT)

ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "ane_bisection")
NUM_WARMUP = 10
NUM_RUNS = 30

from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
    apply_rotary_pos_emb_single, _repeat_kv,
)


def ane_conv_state_shape(conv_dim, conv_kernel):
    """Match the ANE-safe shape computation from the converter."""
    total = conv_dim * conv_kernel
    # Find approximately square factorization
    import math
    sqrt = int(math.sqrt(total))
    for d in range(sqrt, 0, -1):
        if total % d == 0:
            return d, total // d
    return 1, total


class FlexChunkWrapper(nn.Module):
    """Wrapper for exporting a slice of Qwen3.5 layers (any mix of F and L).

    - F layers: KV cache as CoreML states (register_buffer)
    - L layers: conv_state + recurrent_state as I/O tensors
    """

    def __init__(self, model, start_layer, end_layer, ctx):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.ctx = ctx
        self.num_layers = end_layer - start_layer
        cfg = model.config

        # Count F and L layers
        self.f_count = 0
        self.l_count = 0
        self.layer_types = []
        for i in range(start_layer, end_layer):
            lt = model.model.layers[i].layer_type
            self.layer_types.append(lt)
            if lt == "full_attention":
                self.f_count += 1
            else:
                self.l_count += 1

        # F-layer KV cache: one per F layer
        f_idx = 0
        self._f_layer_map = {}  # local_idx → f_cache_idx
        for local_idx in range(self.num_layers):
            if self.layer_types[local_idx] == "full_attention":
                self.register_buffer(f"k_cache_{f_idx}", torch.zeros(
                    1, cfg.num_key_value_heads, ctx, cfg.head_dim,
                    dtype=MODEL_DTYPE, device=TEST_DEVICE))
                self.register_buffer(f"v_cache_{f_idx}", torch.zeros(
                    1, cfg.num_key_value_heads, ctx, cfg.head_dim,
                    dtype=MODEL_DTYPE, device=TEST_DEVICE))
                self._f_layer_map[local_idx] = f_idx
                f_idx += 1

        # L-layer state shapes (for I/O tensors)
        self._has_linear = self.l_count > 0
        if self._has_linear:
            tc = cfg.text_config
            conv_dim = (tc.linear_num_key_heads * tc.linear_key_head_dim * 2
                        + tc.linear_num_value_heads * tc.linear_value_head_dim)
            conv_kernel = max(1, int(tc.linear_conv_kernel_dim))
            ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
            self._lin_conv_shape = (self.l_count, ane_dim1, ane_dim2)
            self._lin_rec_shape = (
                self.l_count,
                tc.linear_num_value_heads,
                tc.linear_key_head_dim,
                tc.linear_value_head_dim,
            )
            self._conv_dim = conv_dim
            self._conv_kernel = conv_kernel

    def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                linear_conv_state=None, linear_recurrent_state=None):
        l_idx = 0
        for local_idx in range(self.num_layers):
            layer_idx = self.start_layer + local_idx
            layer = self.model.model.layers[layer_idx]

            if self.layer_types[local_idx] == "full_attention":
                # F-layer forward
                x = layer.input_layernorm(hidden_states)
                query_states, key_states, value_states, gate = layer.self_attn.get_new_kv_cache(
                    x, position_ids)

                f_cache_idx = self._f_layer_map[local_idx]
                k_cache = getattr(self, f"k_cache_{f_cache_idx}")
                v_cache = getattr(self, f"v_cache_{f_cache_idx}")

                pos = current_pos[0]
                k_cache[:, :, pos:pos+1, :] = key_states.squeeze(0)
                v_cache[:, :, pos:pos+1, :] = value_states.squeeze(0)

                attn_out = layer.self_attn.forward_regular(
                    hidden_states=x,
                    query_states=query_states,
                    kv_cache_layer=(k_cache.squeeze(0), v_cache.squeeze(0)),
                    causal_mask=causal_mask,
                    gate=gate,
                )
                hidden_states = hidden_states + attn_out
                post = layer.post_attention_layernorm(hidden_states)
                hidden_states = hidden_states + layer.mlp(post)

            else:
                # L-layer forward
                x = layer.input_layernorm(hidden_states)
                conv_state_flat = linear_conv_state[l_idx:l_idx+1]
                conv_state = conv_state_flat.reshape(1, self._conv_dim, self._conv_kernel)
                recurrent_state = linear_recurrent_state[l_idx:l_idx+1]

                attn_out, next_conv, next_rec = layer.self_attn.forward_regular(
                    hidden_states=x,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                    has_previous_state=True,
                    causal_mask=causal_mask,
                    expected_batch_size=1,
                    expected_seq_len=1,
                    force_recurrent=True,
                )

                ane_dim1, ane_dim2 = self._lin_conv_shape[1], self._lin_conv_shape[2]
                linear_conv_state[l_idx:l_idx+1] = next_conv.reshape(1, ane_dim1, ane_dim2)
                linear_recurrent_state[l_idx:l_idx+1] = next_rec.to(linear_recurrent_state.dtype)

                hidden_states = hidden_states + attn_out
                post = layer.post_attention_layernorm(hidden_states)
                hidden_states = hidden_states + layer.mlp(post)
                l_idx += 1

        if self._has_linear:
            return hidden_states, linear_conv_state, linear_recurrent_state
        return hidden_states


def export_chunk(model, start_layer, end_layer, ctx, skip_existing=False):
    """Export a chunk of layers."""
    wrapper = FlexChunkWrapper(model, start_layer, end_layer, ctx)
    wrapper.eval()

    cfg = model.config
    layer_types = wrapper.layer_types
    type_str = "".join("F" if t == "full_attention" else "L" for t in layer_types)
    name = f"chunk_{type_str}_L{start_layer}-{end_layer}_ctx{ctx}"
    out_path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")

    if skip_existing and os.path.exists(out_path):
        print(f"  [skip] {name}")
        return out_path, name, wrapper

    # Sample inputs
    hidden = torch.randn(1, 1, cfg.hidden_size, dtype=torch.float32, device=TEST_DEVICE)
    pos_ids = torch.tensor([ctx // 2], dtype=torch.long, device=TEST_DEVICE)
    mask = torch.zeros(1, 1, 1, ctx, dtype=torch.float32, device=TEST_DEVICE)
    cur_pos = torch.tensor([ctx // 2], dtype=torch.int32, device=TEST_DEVICE)

    args = [hidden, pos_ids, mask, cur_pos]

    has_linear = wrapper._has_linear
    if has_linear:
        lin_conv = torch.zeros(*wrapper._lin_conv_shape, dtype=torch.float32, device=TEST_DEVICE)
        lin_rec = torch.zeros(*wrapper._lin_rec_shape, dtype=torch.float32, device=TEST_DEVICE)
        args.extend([lin_conv, lin_rec])

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, tuple(args))

    # CoreML inputs
    ct_inputs = [
        ct.TensorType(name="hidden_states", shape=hidden.shape, dtype=np.float16),
        ct.TensorType(name="position_ids", shape=pos_ids.shape, dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=mask.shape, dtype=np.float16),
        ct.TensorType(name="current_pos", shape=cur_pos.shape, dtype=np.int32),
    ]
    if has_linear:
        ct_inputs.append(ct.TensorType(name="linear_conv_state",
                                        shape=lin_conv.shape, dtype=np.float16))
        ct_inputs.append(ct.TensorType(name="linear_recurrent_state",
                                        shape=lin_rec.shape, dtype=np.float16))

    # CoreML outputs
    ct_outputs = [ct.TensorType(name="output", dtype=np.float16)]
    if has_linear:
        ct_outputs.append(ct.TensorType(name="linear_conv_state_out", dtype=np.float16))
        ct_outputs.append(ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16))

    # CoreML states (F-layer KV caches)
    states = []
    for i in range(wrapper.f_count):
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, cfg.num_key_value_heads, ctx, cfg.head_dim),
                dtype=np.float16),
            name=f"k_cache_{i}"))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, cfg.num_key_value_heads, ctx, cfg.head_dim),
                dtype=np.float16),
            name=f"v_cache_{i}"))

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        states=states if states else None,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mlmodel.save(out_path)
    del mlmodel, traced
    gc.collect()
    return out_path, name, wrapper


def analyze_mil(path, name):
    mlmodel = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    spec = mlmodel.get_spec()
    prog = spec.mlProgram
    op_counts = {}
    hostile = []
    for fn in prog.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                t = op.type
                op_counts[t] = op_counts.get(t, 0) + 1
                if t in ("gather", "scatter", "greater_equal", "select",
                         "read_state", "coreml_update_state"):
                    hostile.append(t)
    total = sum(op_counts.values())
    conv = op_counts.get("conv", 0)
    matmul = op_counts.get("matmul", 0) + op_counts.get("einsum", 0)
    softmax = op_counts.get("softmax", 0)
    hostile_str = ", ".join(f"{h}({hostile.count(h)})" for h in sorted(set(hostile))) if hostile else "NONE"
    print(f"  {name:50s} ops={total:4d} conv={conv:2d} mm={matmul:2d} sm={softmax:2d} hostile={hostile_str}")
    del mlmodel
    return total


def measure_ane(path, name, has_state, has_linear, wrapper_info, ctx,
                compute_unit=ct.ComputeUnit.CPU_AND_NE):
    mlmodel = ct.models.MLModel(path, compute_units=compute_unit)

    cfg_hidden = 2560  # hardcoded for simplicity
    pred = {
        "hidden_states": np.random.randn(1, 1, cfg_hidden).astype(np.float16),
        "position_ids": np.array([ctx // 2], dtype=np.int32),
        "causal_mask": np.zeros((1, 1, 1, ctx), dtype=np.float16),
        "current_pos": np.array([ctx // 2], dtype=np.int32),
    }
    if has_linear:
        pred["linear_conv_state"] = np.zeros(wrapper_info["conv_shape"], dtype=np.float16)
        pred["linear_recurrent_state"] = np.zeros(wrapper_info["rec_shape"], dtype=np.float16)

    state = mlmodel.make_state() if has_state else None

    for _ in range(NUM_WARMUP):
        if state is not None:
            mlmodel.predict(pred, state=state)
        else:
            mlmodel.predict(pred)

    import resource
    times = []
    cpu_times = []
    for _ in range(NUM_RUNS):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        if state is not None:
            mlmodel.predict(pred, state=state)
        else:
            mlmodel.predict(pred)
        t1 = time.perf_counter()
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        times.append(t1 - t0)
        cpu_times.append((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime))

    wall_ms = np.median(times) * 1000
    cpu_ms = np.median(cpu_times) * 1000
    cpu_pct = (cpu_ms / wall_ms * 100) if wall_ms > 0 else 0
    ane_pct = max(0, 100 - cpu_pct)
    tag = "ANE" if compute_unit == ct.ComputeUnit.CPU_AND_NE else "CPU"
    print(f"  [{tag}] {name:47s} wall={wall_ms:8.2f}ms cpu={cpu_ms:8.2f}ms  CPU%={cpu_pct:5.1f}%  ANE%={ane_pct:5.1f}%")
    del mlmodel
    gc.collect()
    return wall_ms, cpu_ms, ane_pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--cpu-ref", action="store_true")
    args = parser.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("=" * 90)
    print("  ANE BISECTION PHASE 3: F + L layer combinations")
    print(f"  CTX: {args.ctx}")
    print("=" * 90)

    # Load model
    HF_MODEL = os.path.join(REPO_ROOT, "models", "Qwen__Qwen3.5-4B")
    print("\nLoading Qwen3.5-4B...")
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    cfg.context_length = max(args.ctx)
    cfg.state_length = max(args.ctx)
    model = Qwen35ForCausalLM(cfg)

    if not args.skip_export:
        assert model.load_pretrained_weights(HF_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Print layer types
    for i, layer in enumerate(model.model.layers):
        lt = "F" if layer.layer_type == "full_attention" else "L"
        if i < 12:
            print(f"  Layer {i:2d}: {lt}")
    print(f"  ... ({len(model.model.layers)} total)")

    # Define test configurations: (start_layer, end_layer, description)
    configs = [
        (0, 1, "1L"),          # Single L layer
        (0, 3, "3L"),          # Three L layers (chunk 0 pattern)
        (3, 4, "1F"),          # Single F layer (baseline)
        (3, 5, "1F+1L"),       # F + 1 L
        (3, 6, "1F+2L"),       # F + 2 L
        (3, 7, "1F+3L"),       # F + 3 L (actual chunk 1 pattern: FLLL)
        (7, 11, "1F+3L_c2"),   # Chunk 2 pattern: FLLL
    ]

    # Export
    exported = {}
    if not args.skip_export:
        print(f"\n{'='*90}")
        print("  EXPORT")
        print(f"{'='*90}")
        for start, end, desc in configs:
            for ctx in args.ctx:
                t0 = time.time()
                try:
                    path, name, wrapper = export_chunk(
                        model, start, end, ctx, skip_existing=args.skip_existing)
                    info = {
                        "has_state": wrapper.f_count > 0,
                        "has_linear": wrapper._has_linear,
                        "conv_shape": wrapper._lin_conv_shape if wrapper._has_linear else None,
                        "rec_shape": wrapper._lin_rec_shape if wrapper._has_linear else None,
                    }
                    exported[(name, ctx)] = info
                    print(f"  {name:55s} {time.time()-t0:6.1f}s  ✓  [{desc}]")
                except Exception as e:
                    print(f"  {desc} ctx={ctx}: FAILED: {e}")
                gc.collect()

    del model
    gc.collect()

    # MIL analysis
    print(f"\n{'='*90}")
    print("  MIL OP ANALYSIS")
    print(f"{'='*90}")
    all_configs = []
    for start, end, desc in configs:
        for ctx in args.ctx:
            layer_types = []
            # Reconstruct type string from config
            cfg2 = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
            # We need the layer types but can't load model again. Use known pattern.
            # Qwen3.5-4B: [LLL, FLLL×7, F] → layers 0-2=L, 3=F, 4-6=L, 7=F, ...
            known_f = {3, 7, 11, 15, 19, 23, 27, 31}
            for i in range(start, end):
                layer_types.append("F" if i in known_f else "L")
            type_str = "".join(layer_types)
            name = f"chunk_{type_str}_L{start}-{end}_ctx{ctx}"
            path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
            if os.path.exists(path):
                try:
                    analyze_mil(path, name)
                    info = exported.get((name, ctx), {
                        "has_state": "F" in type_str,
                        "has_linear": "L" in type_str,
                    })
                    all_configs.append((name, ctx, type_str, desc, info))
                except Exception as e:
                    print(f"  {name:50s} FAILED: {e}")

    # ANE measurement
    print(f"\n{'='*90}")
    print(f"  ANE UTILIZATION ({NUM_RUNS} runs, {NUM_WARMUP} warmup)")
    print(f"{'='*90}")
    results = {}

    for ctx in args.ctx:
        print(f"\n  --- CTX={ctx} ---")
        for name, c, type_str, desc, info in all_configs:
            if c != ctx:
                continue
            path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
            if not os.path.exists(path):
                continue

            # Reconstruct wrapper info for prediction
            known_f = {3, 7, 11, 15, 19, 23, 27, 31}
            has_state = any(ch == "F" for ch in type_str)
            has_linear = any(ch == "L" for ch in type_str)

            if has_linear and "conv_shape" not in info:
                # Compute from config
                tc = cfg2.text_config if hasattr(cfg2, 'text_config') else None
                if tc is None:
                    cfg2 = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
                    tc = cfg2.text_config
                conv_dim = (tc.linear_num_key_heads * tc.linear_key_head_dim * 2
                            + tc.linear_num_value_heads * tc.linear_value_head_dim)
                conv_kernel = max(1, int(tc.linear_conv_kernel_dim))
                ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
                l_count = type_str.count("L")
                info["conv_shape"] = (l_count, ane_dim1, ane_dim2)
                info["rec_shape"] = (l_count, tc.linear_num_value_heads,
                                     tc.linear_key_head_dim, tc.linear_value_head_dim)

            try:
                wall, cpu, ane = measure_ane(
                    path, name, has_state, has_linear, info, ctx)
                results[(name, ctx)] = (wall, cpu, ane, desc)
            except Exception as e:
                print(f"  {name:55s} FAILED: {e}")
            gc.collect()

        if args.cpu_ref:
            print(f"\n  --- CTX={ctx} CPU_ONLY ---")
            for name, c, type_str, desc, info in all_configs:
                if c != ctx:
                    continue
                path = os.path.join(ARTIFACT_DIR, f"{name}.mlpackage")
                if not os.path.exists(path):
                    continue
                has_state = any(ch == "F" for ch in type_str)
                has_linear = any(ch == "L" for ch in type_str)
                try:
                    measure_ane(path, name, has_state, has_linear, info, ctx,
                               ct.ComputeUnit.CPU_ONLY)
                except Exception as e:
                    pass
                gc.collect()

    # Summary
    print(f"\n{'='*90}")
    print("  SUMMARY: F + L layer combinations")
    print(f"{'='*90}")
    header = f"  {'Config':<25s} {'Pattern':<8s}"
    for ctx in args.ctx:
        header += f"  {'CTX='+str(ctx):>16s}"
    print(header)
    print("  " + "-" * (25 + 8 + 18 * len(args.ctx)))

    for start, end, desc in configs:
        known_f = {3, 7, 11, 15, 19, 23, 27, 31}
        type_str = "".join("F" if i in known_f else "L" for i in range(start, end))
        row = f"  {desc:<25s} {type_str:<8s}"
        for ctx in args.ctx:
            name = f"chunk_{type_str}_L{start}-{end}_ctx{ctx}"
            key = (name, ctx)
            if key in results:
                wall, cpu, ane, _ = results[key]
                row += f"  {ane:7.1f}% {wall:6.1f}ms"
            else:
                row += f"  {'---':>16s}"
        print(row)

    # Transition analysis
    print(f"\n  ANE transition (incremental L layers added to F):")
    for ctx in args.ctx:
        print(f"  CTX={ctx}:")
        prev_ane = None
        for start, end, desc in configs:
            known_f = {3, 7, 11, 15, 19, 23, 27, 31}
            type_str = "".join("F" if i in known_f else "L" for i in range(start, end))
            name = f"chunk_{type_str}_L{start}-{end}_ctx{ctx}"
            key = (name, ctx)
            if key not in results:
                continue
            _, _, ane, _ = results[key]
            if prev_ane is not None:
                delta = ane - prev_ane
                marker = ""
                if delta < -15:
                    marker = " <<<< SIGNIFICANT DROP"
                if delta < -30:
                    marker = " <<<< ANE KILLED"
                print(f"    {desc:<20s} {type_str:<8s} ANE={ane:5.1f}%  Δ={delta:+6.1f}%{marker}")
            else:
                print(f"    {desc:<20s} {type_str:<8s} ANE={ane:5.1f}%  (first)")
            prev_ane = ane

    print(f"\n{'='*90}")
    print("  Done!")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
