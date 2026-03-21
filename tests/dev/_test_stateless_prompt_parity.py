#!/usr/bin/env python3
"""Prompt-level parity: stateful vs stateless linear attention state.

Exports a single FFN chunk (chunk 0, layers 0-7) in two variants:
  1. STATEFUL — linear_conv_state and linear_recurrent_state as ct.StateType
  2. STATELESS — linear states as regular input/output tensors

Then runs full prompt pipeline (prefill + decode) comparing:
  - PyTorch reference (fp32 recurrence)
  - Stateful CoreML (current design)
  - Stateless CoreML (proposed fix)

Measures hidden-state cosine per token and final text output.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, argparse, time, tempfile
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    MODEL_DTYPE, TEST_DEVICE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
CTX = 256
CHUNK_IDX = 0
TOTAL_CHUNKS = 4
LAYERS_PER_CHUNK = 8
START_LAYER = 0
END_LAYER = 8


def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


# ── Stateless FFN Wrapper ──
class StatelessFFNWrapper(nn.Module):
    """FFN wrapper where linear attention states are regular I/O, not buffers.
    
    KV cache for full attention stays as register_buffer (CoreML state)
    since those grow with context and are much larger.
    Linear attention states become explicit inputs/outputs.
    """
    def __init__(self, model, start_layer, end_layer):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.local_num_layers = end_layer - start_layer
        cfg = model.config
        # KV cache stays as state (large, context-dependent)
        self.register_buffer("k_cache", torch.zeros(
            self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim,
            dtype=MODEL_DTYPE, device=TEST_DEVICE))
        self.register_buffer("v_cache", torch.zeros(
            self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim,
            dtype=MODEL_DTYPE, device=TEST_DEVICE))
    
    def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                linear_conv_state, linear_recurrent_state):
        out = self.model.model.process_layers_regular_single_token_export_local_state(
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
        )
        return out, linear_conv_state, linear_recurrent_state


def export_stateful(model, cfg, tmpdir):
    """Export chunk 0 with stateful linear attention (current design)."""
    print("  Exporting stateful model...")
    converter = Qwen35Converter(model, context_length=CTX, batch_size=1, num_chunks=TOTAL_CHUNKS)
    mlmodel = converter.convert_part_2(model, chunk_idx=CHUNK_IDX, total_chunks=TOTAL_CHUNKS)
    path = os.path.join(tmpdir, "stateful.mlpackage")
    mlmodel.save(path)
    del mlmodel, converter; gc.collect()
    return path


def export_stateless(model, cfg, tmpdir):
    """Export chunk 0 with stateless linear attention."""
    print("  Exporting stateless model...")
    conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)
    
    wrapper = StatelessFFNWrapper(model, START_LAYER, END_LAYER).eval()
    
    hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=MODEL_DTYPE)
    position_ids = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, 1, CTX), dtype=MODEL_DTYPE)
    current_pos = torch.zeros((1,), dtype=torch.int32)
    lin_conv = torch.zeros((LAYERS_PER_CHUNK, ane_d1, ane_d2), dtype=MODEL_DTYPE)
    lin_rec = torch.zeros((LAYERS_PER_CHUNK, cfg.text_config.linear_num_value_heads,
                           cfg.text_config.linear_key_head_dim,
                           cfg.text_config.linear_value_head_dim), dtype=MODEL_DTYPE)
    
    # Reset state
    with torch.no_grad():
        wrapper.k_cache.zero_()
        wrapper.v_cache.zero_()
    
    traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos,
                                       lin_conv, lin_rec), check_trace=False)
    with torch.no_grad():
        traced.k_cache.zero_()
        traced.v_cache.zero_()
    
    # KV cache stays as state
    kv_states = [
        ct.StateType(wrapped_type=ct.TensorType(
            shape=(LAYERS_PER_CHUNK, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
            dtype=np.float16), name="k_cache"),
        ct.StateType(wrapped_type=ct.TensorType(
            shape=(LAYERS_PER_CHUNK, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
            dtype=np.float16), name="v_cache"),
    ]
    
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
        ],
        states=kv_states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    path = os.path.join(tmpdir, "stateless.mlpackage")
    mlmodel.save(path)
    del mlmodel, wrapper, traced; gc.collect()
    return path


def run_pytorch(model, cfg, ids, prompt_len, max_gen, tokenizer):
    """Run PyTorch reference through chunk 0 only."""
    conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)
    
    k_cache = torch.zeros(LAYERS_PER_CHUNK, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
    v_cache = torch.zeros(LAYERS_PER_CHUNK, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
    conv_state = torch.zeros(LAYERS_PER_CHUNK, ane_d1, ane_d2, dtype=MODEL_DTYPE)
    rec_state = torch.zeros(LAYERS_PER_CHUNK, cfg.text_config.linear_num_value_heads,
                            cfg.text_config.linear_key_head_dim,
                            cfg.text_config.linear_value_head_dim, dtype=MODEL_DTYPE)
    
    hiddens = []
    with torch.no_grad():
        for pos in range(prompt_len + max_gen):
            if pos < prompt_len:
                tok = ids[:, pos:pos+1].to(torch.int32)
            else:
                tok = next_id
            hidden = model.model.embed_tokens(tok).to(MODEL_DTYPE)
            mask = torch.full((1, 1, 1, CTX), float("-inf"), dtype=MODEL_DTYPE)
            mask[:, :, :, :pos+1] = 0
            position_ids = torch.tensor([pos], dtype=torch.int32)
            current_pos = torch.tensor([pos], dtype=torch.int32)
            
            hidden = model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=hidden, position_ids=position_ids,
                causal_mask=mask, current_pos=current_pos,
                kv_cache_0=None, k_cache=k_cache, v_cache=v_cache,
                linear_conv_state=conv_state, linear_recurrent_state=rec_state,
                start_layer=START_LAYER, end_layer=END_LAYER,
                apply_final_norm=False,
            )
            # Apply final norm (only chunk 0, but let's include it for lm_head)
            hidden_normed = model.model.norm(hidden)
            logits = model.lm_head(hidden_normed.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)
            next_id = torch.argmax(logits, dim=-1).to(torch.int32)
            
            hiddens.append(hidden.numpy().copy())
            if pos % 20 == 0:
                sys.stdout.write(f"\r  PT: {pos+1}/{prompt_len+max_gen}")
                sys.stdout.flush()
    print()
    return hiddens


def run_coreml_stateful(path, model, cfg, ids, prompt_len, max_gen, tokenizer, compute_unit):
    """Run CoreML stateful model through chunk 0."""
    cml = ct.models.MLModel(path, compute_units=compute_unit)
    state = cml.make_state()
    
    hiddens = []
    with torch.no_grad():
        for pos in range(prompt_len + max_gen):
            if pos < prompt_len:
                tok = ids[:, pos:pos+1].numpy().astype(np.int32)
            else:
                tok = np.array([[next_id_val]], dtype=np.int32)
            embed = model.model.embed_tokens(torch.tensor(tok)).to(MODEL_DTYPE).numpy()
            mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
            mask[:, :, :, :pos+1] = 0
            
            out = cml.predict({
                "hidden_states": embed.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
            }, state=state)
            hidden = out["output_hidden_states"]
            hiddens.append(hidden.copy())
            
            # LM head in PyTorch for fair comparison
            h_t = torch.tensor(hidden).to(MODEL_DTYPE)
            h_normed = model.model.norm(h_t)
            logits = model.lm_head(h_normed.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)
            next_id_val = int(torch.argmax(logits).item())
            
            if pos % 20 == 0:
                sys.stdout.write(f"\r  Stateful: {pos+1}/{prompt_len+max_gen}")
                sys.stdout.flush()
    print()
    del cml, state; gc.collect()
    return hiddens


def run_coreml_stateless(path, model, cfg, ids, prompt_len, max_gen, tokenizer, compute_unit):
    """Run CoreML stateless model through chunk 0."""
    conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)
    
    cml = ct.models.MLModel(path, compute_units=compute_unit)
    state = cml.make_state()  # for KV cache only
    
    # Linear states as regular numpy arrays
    lin_conv = np.zeros((LAYERS_PER_CHUNK, ane_d1, ane_d2), dtype=np.float16)
    lin_rec = np.zeros((LAYERS_PER_CHUNK, cfg.text_config.linear_num_value_heads,
                        cfg.text_config.linear_key_head_dim,
                        cfg.text_config.linear_value_head_dim), dtype=np.float16)
    
    hiddens = []
    with torch.no_grad():
        for pos in range(prompt_len + max_gen):
            if pos < prompt_len:
                tok = ids[:, pos:pos+1].numpy().astype(np.int32)
            else:
                tok = np.array([[next_id_val]], dtype=np.int32)
            embed = model.model.embed_tokens(torch.tensor(tok)).to(MODEL_DTYPE).numpy()
            mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
            mask[:, :, :, :pos+1] = 0
            
            out = cml.predict({
                "hidden_states": embed.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
                "linear_conv_state": lin_conv,
                "linear_recurrent_state": lin_rec,
            }, state=state)
            hidden = out["output_hidden_states"]
            lin_conv = out["linear_conv_state_out"]
            lin_rec = out["linear_recurrent_state_out"]
            hiddens.append(hidden.copy())
            
            # LM head in PyTorch for fair comparison
            h_t = torch.tensor(hidden).to(MODEL_DTYPE)
            h_normed = model.model.norm(h_t)
            logits = model.lm_head(h_normed.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)
            next_id_val = int(torch.argmax(logits).item())
            
            if pos % 20 == 0:
                sys.stdout.write(f"\r  Stateless: {pos+1}/{prompt_len+max_gen}")
                sys.stdout.flush()
    print()
    del cml, state; gc.collect()
    return hiddens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=30, help="decode tokens")
    parser.add_argument("--prompt", type=str,
                        default="Explain the difference between a stack and a queue in computer science.",
                        help="Input prompt")
    parser.add_argument("--cpu-only", action="store_true", help="CPU only (no ANE)")
    args = parser.parse_args()
    max_gen = args.tokens
    compute_unit = ct.ComputeUnit.CPU_ONLY if args.cpu_only else ct.ComputeUnit.CPU_AND_NE
    tmpdir = tempfile.mkdtemp(prefix="stateless_prompt_")
    
    print("=" * 70)
    print("  Prompt-Level Parity: Stateful vs Stateless Linear Attention")
    print(f"  Chunk 0 (layers {START_LAYER}-{END_LAYER}), CTX={CTX}")
    print(f"  Compute: {compute_unit}")
    print(f"  Temp: {tmpdir}")
    print("=" * 70)
    
    # ── 1. Load model & tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    input_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True).input_ids
    prompt_len = input_ids.shape[1]
    if prompt_len >= CTX - max_gen:
        input_ids = input_ids[:, :CTX - max_gen]
        prompt_len = input_ids.shape[1]
    print(f"Prompt: {prompt_len} tokens, decode: {max_gen} tokens")
    
    print("\nLoading model...")
    cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(MODEL_PATH), "weight load failed"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    # ── 2. Export both variants ──
    print("\n--- Exporting Models ---")
    sf_path = export_stateful(model, cfg, tmpdir)
    sl_path = export_stateless(model, cfg, tmpdir)
    
    # ── 3. Run all three ──
    print("\n--- Running PyTorch Reference ---")
    pt_hiddens = run_pytorch(model, cfg, input_ids, prompt_len, max_gen, tokenizer)
    
    print("\n--- Running CoreML Stateful ---")
    sf_hiddens = run_coreml_stateful(sf_path, model, cfg, input_ids, prompt_len, max_gen, tokenizer, compute_unit)
    
    print("\n--- Running CoreML Stateless ---")
    sl_hiddens = run_coreml_stateless(sl_path, model, cfg, input_ids, prompt_len, max_gen, tokenizer, compute_unit)
    
    # ── 4. Compare ──
    total_steps = prompt_len + max_gen
    
    print(f"\n{'='*70}")
    print(f"  RESULTS: Hidden State Cosine Similarity (vs PyTorch)")
    print(f"{'='*70}")
    print(f"\n{'':>6} {'Phase':<10} {'Stateful':>14} {'Stateless':>14} {'Better':>10}")
    print("-" * 58)
    
    sf_cos_prefill = []
    sl_cos_prefill = []
    sf_cos_decode = []
    sl_cos_decode = []
    
    for pos in range(total_steps):
        phase = "prefill" if pos < prompt_len else "decode"
        cos_sf = cosine(sf_hiddens[pos], pt_hiddens[pos])
        cos_sl = cosine(sl_hiddens[pos], pt_hiddens[pos])
        
        if phase == "prefill":
            sf_cos_prefill.append(cos_sf)
            sl_cos_prefill.append(cos_sl)
        else:
            sf_cos_decode.append(cos_sf)
            sl_cos_decode.append(cos_sl)
        
        better = "stateless" if cos_sl > cos_sf + 0.001 else ("stateful" if cos_sf > cos_sl + 0.001 else "tie")
        
        # Print every step for decode, every 10 for prefill
        if phase == "decode" or pos % 10 == 0 or pos == prompt_len - 1:
            marker = " ✓" if better == "stateless" else ""
            print(f"  {pos:4d} {phase:<10} {cos_sf:14.8f} {cos_sl:14.8f} {better:>10}{marker}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<30} {'Stateful':>14} {'Stateless':>14}")
    print("-" * 62)
    if sf_cos_prefill:
        print(f"{'Prefill avg cos':<30} {np.mean(sf_cos_prefill):14.8f} {np.mean(sl_cos_prefill):14.8f}")
        print(f"{'Prefill min cos':<30} {np.min(sf_cos_prefill):14.8f} {np.min(sl_cos_prefill):14.8f}")
    if sf_cos_decode:
        print(f"{'Decode avg cos':<30} {np.mean(sf_cos_decode):14.8f} {np.mean(sl_cos_decode):14.8f}")
        print(f"{'Decode min cos':<30} {np.min(sf_cos_decode):14.8f} {np.min(sl_cos_decode):14.8f}")
    
    all_sf = sf_cos_prefill + sf_cos_decode
    all_sl = sl_cos_prefill + sl_cos_decode
    print(f"{'Overall avg cos':<30} {np.mean(all_sf):14.8f} {np.mean(all_sl):14.8f}")
    print(f"{'Overall min cos':<30} {np.min(all_sf):14.8f} {np.min(all_sl):14.8f}")
    
    improvement = np.mean(all_sl) - np.mean(all_sf)
    if improvement > 0.001:
        print(f"\n✅ Stateless improves parity by +{improvement:.6f} avg cosine")
    elif improvement < -0.001:
        print(f"\n❌ Stateless is WORSE by {improvement:.6f} avg cosine")
    else:
        print(f"\n≈ Stateful and stateless are equivalent ({improvement:+.6f})")
    
    print(f"\nCleanup: rm -rf {tmpdir}")


if __name__ == "__main__":
    main()
