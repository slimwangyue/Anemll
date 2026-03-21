#!/usr/bin/env python3
"""Teacher-forced decode parity: stateful vs stateless linear attention.

KEY INSIGHT from previous test:
  The autoregressive test gave ~0.5 decode cosine because once the first
  token diverges, each path generates different tokens and the comparison
  becomes meaningless. 

This test uses TEACHER FORCING: feed the SAME tokens to all three paths
(PyTorch, stateful CoreML, stateless CoreML) at every step. This isolates
the per-step model divergence without feedback amplification.

Exports chunk 0 (layers 0-7) in both stateful and stateless variants.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, argparse, tempfile
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


class StatelessFFNWrapper(nn.Module):
    """FFN wrapper: linear attention states as I/O, KV cache as state."""
    def __init__(self, model, start_layer, end_layer):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.local_num_layers = end_layer - start_layer
        cfg = model.config
        self.register_buffer("k_cache", torch.zeros(
            self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim,
            dtype=MODEL_DTYPE))
        self.register_buffer("v_cache", torch.zeros(
            self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim,
            dtype=MODEL_DTYPE))

    def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                linear_conv_state, linear_recurrent_state):
        out = self.model.model.process_layers_regular_single_token_export_local_state(
            hidden_states=hidden_states, position_ids=position_ids,
            causal_mask=causal_mask, current_pos=current_pos,
            kv_cache_0=None, k_cache=self.k_cache, v_cache=self.v_cache,
            linear_conv_state=linear_conv_state,
            linear_recurrent_state=linear_recurrent_state,
            start_layer=self.start_layer, end_layer=self.end_layer,
            apply_final_norm=False,
        )
        return out, linear_conv_state, linear_recurrent_state


def export_stateful(model, cfg, tmpdir):
    print("  Exporting stateful chunk 0...")
    converter = Qwen35Converter(model, context_length=CTX, batch_size=1, num_chunks=TOTAL_CHUNKS)
    mlmodel = converter.convert_part_2(model, chunk_idx=CHUNK_IDX, total_chunks=TOTAL_CHUNKS)
    path = os.path.join(tmpdir, "stateful.mlpackage")
    mlmodel.save(path)
    del mlmodel, converter; gc.collect()
    return path


def export_stateless(model, cfg, tmpdir):
    print("  Exporting stateless chunk 0...")
    conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)

    wrapper = StatelessFFNWrapper(model, START_LAYER, END_LAYER).eval()
    h = torch.zeros((1, 1, cfg.hidden_size), dtype=MODEL_DTYPE)
    pos = torch.zeros((1,), dtype=torch.int32)
    mask = torch.zeros((1, 1, 1, CTX), dtype=MODEL_DTYPE)
    cpos = torch.zeros((1,), dtype=torch.int32)
    lc = torch.zeros((LAYERS_PER_CHUNK, ane_d1, ane_d2), dtype=MODEL_DTYPE)
    lr = torch.zeros((LAYERS_PER_CHUNK, cfg.text_config.linear_num_value_heads,
                      cfg.text_config.linear_key_head_dim,
                      cfg.text_config.linear_value_head_dim), dtype=MODEL_DTYPE)

    with torch.no_grad():
        wrapper.k_cache.zero_(); wrapper.v_cache.zero_()
    traced = torch.jit.trace(wrapper, (h, pos, mask, cpos, lc, lr), check_trace=False)
    with torch.no_grad():
        traced.k_cache.zero_(); traced.v_cache.zero_()

    kv_states = [
        ct.StateType(wrapped_type=ct.TensorType(
            shape=(LAYERS_PER_CHUNK, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
            dtype=np.float16), name="k_cache"),
        ct.StateType(wrapped_type=ct.TensorType(
            shape=(LAYERS_PER_CHUNK, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
            dtype=np.float16), name="v_cache"),
    ]
    mlmodel = ct.convert(traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=h.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=pos.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=cpos.shape, dtype=np.int32),
            ct.TensorType(name="linear_conv_state", shape=lc.shape, dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state", shape=lr.shape, dtype=np.float16),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=50, help="total tokens to process")
    parser.add_argument("--prompt", type=str,
                        default="Explain the difference between a stack and a queue in computer science.",
                        help="Input prompt")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    compute_unit = ct.ComputeUnit.CPU_ONLY if args.cpu_only else ct.ComputeUnit.CPU_AND_NE
    tmpdir = tempfile.mkdtemp(prefix="teacher_forced_")

    print("=" * 70)
    print("  Teacher-Forced Decode Parity: Stateful vs Stateless")
    print(f"  Chunk 0 (layers {START_LAYER}-{END_LAYER}), CTX={CTX}")
    print(f"  Compute: {compute_unit}")
    print(f"  Temp: {tmpdir}")
    print("=" * 70)

    # ── 1. Load model & tokenize ──
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    prompt = args.prompt
    # Pad or repeat prompt to get desired token count
    while True:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        if ids.shape[1] >= args.tokens:
            ids = ids[:, :args.tokens]
            break
        prompt = prompt + " " + args.prompt
    total_tokens = ids.shape[1]
    print(f"Processing {total_tokens} tokens (teacher-forced, same input to all)")

    cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(MODEL_PATH), "weight load failed"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_d1, ane_d2 = ane_conv_state_shape(conv_dim, conv_kernel)

    # ── 2. Export ──
    print("\n--- Exporting ---")
    sf_path = export_stateful(model, cfg, tmpdir)
    sl_path = export_stateless(model, cfg, tmpdir)

    # ── 3. Pre-compute embeddings (shared across all paths) ──
    print("\nPre-computing embeddings...")
    embeddings = []
    with torch.no_grad():
        for pos in range(total_tokens):
            tok = ids[:, pos:pos+1].to(torch.int32)
            emb = model.model.embed_tokens(tok).to(MODEL_DTYPE)
            embeddings.append(emb)

    # ── 4. PyTorch reference ──
    print("\n--- PyTorch (teacher-forced) ---")
    pt_k = torch.zeros(LAYERS_PER_CHUNK, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
    pt_v = torch.zeros(LAYERS_PER_CHUNK, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
    pt_conv = torch.zeros(LAYERS_PER_CHUNK, ane_d1, ane_d2, dtype=MODEL_DTYPE)
    pt_rec = torch.zeros(LAYERS_PER_CHUNK, cfg.text_config.linear_num_value_heads,
                         cfg.text_config.linear_key_head_dim,
                         cfg.text_config.linear_value_head_dim, dtype=MODEL_DTYPE)
    pt_hiddens = []
    with torch.no_grad():
        for pos in range(total_tokens):
            mask = torch.full((1, 1, 1, CTX), float("-inf"), dtype=MODEL_DTYPE)
            mask[:, :, :, :pos+1] = 0
            hidden = model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=embeddings[pos], position_ids=torch.tensor([pos], dtype=torch.int32),
                causal_mask=mask, current_pos=torch.tensor([pos], dtype=torch.int32),
                kv_cache_0=None, k_cache=pt_k, v_cache=pt_v,
                linear_conv_state=pt_conv, linear_recurrent_state=pt_rec,
                start_layer=START_LAYER, end_layer=END_LAYER,
                apply_final_norm=False,
            )
            pt_hiddens.append(hidden.numpy().copy())
            if pos % 10 == 0:
                sys.stdout.write(f"\r  PT: {pos+1}/{total_tokens}")
                sys.stdout.flush()
    print()

    # ── 5. CoreML stateful (same tokens) ──
    print("\n--- CoreML Stateful (teacher-forced) ---")
    cml_sf = ct.models.MLModel(sf_path, compute_units=compute_unit)
    sf_state = cml_sf.make_state()
    sf_hiddens = []
    for pos in range(total_tokens):
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos+1] = 0
        out = cml_sf.predict({
            "hidden_states": embeddings[pos].numpy().astype(np.float16),
            "position_ids": np.array([pos], dtype=np.int32),
            "causal_mask": mask,
            "current_pos": np.array([pos], dtype=np.int32),
        }, state=sf_state)
        sf_hiddens.append(out["output_hidden_states"].copy())
        if pos % 10 == 0:
            sys.stdout.write(f"\r  Stateful: {pos+1}/{total_tokens}")
            sys.stdout.flush()
    print()
    del cml_sf, sf_state; gc.collect()

    # ── 6. CoreML stateless (same tokens) ──
    print("\n--- CoreML Stateless (teacher-forced) ---")
    cml_sl = ct.models.MLModel(sl_path, compute_units=compute_unit)
    sl_state = cml_sl.make_state()  # KV cache only
    lin_conv = np.zeros((LAYERS_PER_CHUNK, ane_d1, ane_d2), dtype=np.float16)
    lin_rec = np.zeros((LAYERS_PER_CHUNK, cfg.text_config.linear_num_value_heads,
                        cfg.text_config.linear_key_head_dim,
                        cfg.text_config.linear_value_head_dim), dtype=np.float16)
    sl_hiddens = []
    for pos in range(total_tokens):
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos+1] = 0
        out = cml_sl.predict({
            "hidden_states": embeddings[pos].numpy().astype(np.float16),
            "position_ids": np.array([pos], dtype=np.int32),
            "causal_mask": mask,
            "current_pos": np.array([pos], dtype=np.int32),
            "linear_conv_state": lin_conv,
            "linear_recurrent_state": lin_rec,
        }, state=sl_state)
        sl_hiddens.append(out["output_hidden_states"].copy())
        lin_conv = out["linear_conv_state_out"]
        lin_rec = out["linear_recurrent_state_out"]
        if pos % 10 == 0:
            sys.stdout.write(f"\r  Stateless: {pos+1}/{total_tokens}")
            sys.stdout.flush()
    print()
    del cml_sl, sl_state; gc.collect()

    # ── 7. Compare: per-token hidden state cosine ──
    print(f"\n{'='*75}")
    print(f"  Teacher-Forced Decode: Hidden State Cosine vs PyTorch")
    print(f"{'='*75}")
    print(f"\n{'Pos':>4} {'Stateful':>14} {'Stateless':>14} {'SF-SL diff':>12} {'Winner':>10}")
    print("-" * 60)

    sf_cos_all = []
    sl_cos_all = []

    for pos in range(total_tokens):
        cos_sf = cosine(sf_hiddens[pos], pt_hiddens[pos])
        cos_sl = cosine(sl_hiddens[pos], pt_hiddens[pos])
        sf_cos_all.append(cos_sf)
        sl_cos_all.append(cos_sl)

        diff = cos_sl - cos_sf
        winner = "stateless" if diff > 0.001 else ("stateful" if diff < -0.001 else "tie")
        marker = " ✓" if winner == "stateless" else ""

        # Print every step for first 20, then every 5
        if pos < 20 or pos % 5 == 0 or pos == total_tokens - 1:
            print(f"  {pos:4d} {cos_sf:14.8f} {cos_sl:14.8f} {diff:+12.8f} {winner:>10}{marker}")

    # Also compare stateful vs stateless directly (without PyTorch reference)
    sf_sl_cos = []
    for pos in range(total_tokens):
        sf_sl_cos.append(cosine(sf_hiddens[pos], sl_hiddens[pos]))

    # ── 8. Summary ──
    print(f"\n{'='*75}")
    print(f"  SUMMARY ({total_tokens} teacher-forced tokens)")
    print(f"{'='*75}")

    print(f"\n{'Metric':<35} {'Stateful':>14} {'Stateless':>14}")
    print("-" * 65)
    print(f"{'Avg cosine vs PT':<35} {np.mean(sf_cos_all):14.8f} {np.mean(sl_cos_all):14.8f}")
    print(f"{'Min cosine vs PT':<35} {np.min(sf_cos_all):14.8f} {np.min(sl_cos_all):14.8f}")
    print(f"{'Cosine at token 0':<35} {sf_cos_all[0]:14.8f} {sl_cos_all[0]:14.8f}")
    if total_tokens > 10:
        print(f"{'Avg cosine tokens 0-9':<35} {np.mean(sf_cos_all[:10]):14.8f} {np.mean(sl_cos_all[:10]):14.8f}")
    if total_tokens > 20:
        print(f"{'Avg cosine tokens 10-19':<35} {np.mean(sf_cos_all[10:20]):14.8f} {np.mean(sl_cos_all[10:20]):14.8f}")
    if total_tokens > 30:
        print(f"{'Avg cosine tokens 20-29':<35} {np.mean(sf_cos_all[20:30]):14.8f} {np.mean(sl_cos_all[20:30]):14.8f}")
    last10 = min(10, total_tokens)
    print(f"{'Avg cosine last 10':<35} {np.mean(sf_cos_all[-last10:]):14.8f} {np.mean(sl_cos_all[-last10:]):14.8f}")

    print(f"\n{'Stateful vs Stateless direct':<35} avg={np.mean(sf_sl_cos):.8f}  min={np.min(sf_sl_cos):.8f}")

    wins_sl = sum(1 for s, l in zip(sf_cos_all, sl_cos_all) if l > s + 0.001)
    wins_sf = sum(1 for s, l in zip(sf_cos_all, sl_cos_all) if s > l + 0.001)
    ties = total_tokens - wins_sl - wins_sf
    print(f"\nWins: stateless={wins_sl}, stateful={wins_sf}, tie={ties}")

    improvement = np.mean(sl_cos_all) - np.mean(sf_cos_all)
    if improvement > 0.001:
        print(f"\n✅ Stateless improves parity by +{improvement:.6f} avg cosine")
    elif improvement < -0.001:
        print(f"\n❌ Stateless is WORSE by {improvement:.6f} avg cosine")
    else:
        print(f"\n≈ Approaches are equivalent ({improvement:+.6f})")

    print(f"\nCleanup: rm -rf {tmpdir}")


if __name__ == "__main__":
    main()
