#!/usr/bin/env python3
"""Compare Stateless+LUT4 vs Stateless+noLUT: latency + generated text.

Exports all 4 FFN chunks in both configs (LUT4 and fp16), then runs
end-to-end text generation via PyTorch reference, CoreML+LUT4, CoreML+noLUT.

Reports per-token latency, total latency, and output answer for each.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, time, argparse, tempfile, shutil
import numpy as np
import torch
import coremltools as ct
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape,
    MODEL_DTYPE, TEST_DEVICE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
CTX = 256
NUM_CHUNKS = 4
CHUNKS = [(0, 8), (8, 16), (16, 24), (24, 32)]
LAYERS_PER_CHUNK = 8


def cosine(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / (d + 1e-12))


def export_all_chunks(model, cfg, out_dir, lut_bits, per_channel=8):
    """Export all 4 FFN chunks + embeddings + lm_head."""
    label = f"LUT{lut_bits}" if lut_bits else "fp16"
    print(f"\n{'='*60}")
    print(f"  Exporting {label} ({NUM_CHUNKS} chunks)")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # Embeddings (only need once, share across configs)
    embed_path = os.path.join(out_dir, "embeddings.mlpackage")
    if not os.path.exists(embed_path):
        print("  Exporting embeddings...")
        conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                               num_chunks=NUM_CHUNKS, lut_bits=None)
        ml = conv.convert_part_1(model)
        ml.save(embed_path)
        del ml, conv; gc.collect()
        print("  Embeddings saved.")

    # LM head (only need once, share across configs)
    lmhead_path = os.path.join(out_dir, "lm_head.mlpackage")
    if not os.path.exists(lmhead_path):
        print("  Exporting lm_head...")
        conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                               num_chunks=NUM_CHUNKS, lut_bits=None)
        ml = conv.convert_part_3(model, argmax_in_model=False)
        ml.save(lmhead_path)
        del ml, conv; gc.collect()
        print("  LM head saved.")

    # FFN chunks
    for ci in range(NUM_CHUNKS):
        chunk_path = os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage")
        if os.path.exists(chunk_path):
            print(f"  Chunk {ci} already exists, skipping.")
            continue
        t0 = time.time()
        print(f"  Exporting chunk {ci} ({label})...")
        conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                               num_chunks=NUM_CHUNKS, lut_bits=lut_bits,
                               per_channel=per_channel)
        ml = conv.convert_part_2(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
        ml.save(chunk_path)
        dt = time.time() - t0
        del ml, conv; gc.collect()
        print(f"  Chunk {ci} saved ({dt:.1f}s)")


def run_pytorch_decode(model, cfg, tokenizer, input_ids, max_gen):
    """Run PyTorch greedy decode through all 4 chunks. Returns tokens + timing."""
    ane_d1, ane_d2 = ane_conv_state_shape(
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim,
        max(1, int(cfg.text_config.linear_conv_kernel_dim)))
    prompt_len = input_ids.shape[1]

    pt_k = [torch.zeros(LAYERS_PER_CHUNK, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
            for _ in range(NUM_CHUNKS)]
    pt_v = [torch.zeros(LAYERS_PER_CHUNK, cfg.num_key_value_heads, CTX, cfg.head_dim, dtype=MODEL_DTYPE)
            for _ in range(NUM_CHUNKS)]
    pt_c = [torch.zeros(LAYERS_PER_CHUNK, ane_d1, ane_d2, dtype=MODEL_DTYPE)
            for _ in range(NUM_CHUNKS)]
    pt_r = [torch.zeros(LAYERS_PER_CHUNK, cfg.text_config.linear_num_value_heads,
                        cfg.text_config.linear_key_head_dim,
                        cfg.text_config.linear_value_head_dim, dtype=MODEL_DTYPE)
            for _ in range(NUM_CHUNKS)]

    tokens = []
    with torch.no_grad():
        # Prefill
        t_prefill_start = time.time()
        for pos in range(prompt_len):
            tok = input_ids[:, pos:pos+1].to(torch.int32)
            hidden = model.model.embed_tokens(tok).to(MODEL_DTYPE)
            mask = torch.full((1, 1, 1, CTX), float("-inf"), dtype=MODEL_DTYPE)
            mask[:, :, :, :pos+1] = 0
            for ci, (s, e) in enumerate(CHUNKS):
                hidden = model.model.process_layers_regular_single_token_export_local_state(
                    hidden_states=hidden, position_ids=torch.tensor([pos], dtype=torch.int32),
                    causal_mask=mask, current_pos=torch.tensor([pos], dtype=torch.int32),
                    kv_cache_0=None, k_cache=pt_k[ci], v_cache=pt_v[ci],
                    linear_conv_state=pt_c[ci], linear_recurrent_state=pt_r[ci],
                    start_layer=s, end_layer=e, apply_final_norm=(ci == NUM_CHUNKS - 1))
        logits = model.lm_head(hidden.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)
        next_id = torch.argmax(logits, dim=-1).to(torch.int32)
        tokens.append(next_id.item())
        t_prefill_end = time.time()

        # Decode
        t_decode_start = time.time()
        per_token_times = []
        for gi in range(max_gen - 1):
            t0 = time.time()
            pos = prompt_len + gi
            hidden = model.model.embed_tokens(next_id).to(MODEL_DTYPE)
            mask = torch.full((1, 1, 1, CTX), float("-inf"), dtype=MODEL_DTYPE)
            mask[:, :, :, :pos+1] = 0
            for ci, (s, e) in enumerate(CHUNKS):
                hidden = model.model.process_layers_regular_single_token_export_local_state(
                    hidden_states=hidden, position_ids=torch.tensor([pos], dtype=torch.int32),
                    causal_mask=mask, current_pos=torch.tensor([pos], dtype=torch.int32),
                    kv_cache_0=None, k_cache=pt_k[ci], v_cache=pt_v[ci],
                    linear_conv_state=pt_c[ci], linear_recurrent_state=pt_r[ci],
                    start_layer=s, end_layer=e, apply_final_norm=(ci == NUM_CHUNKS - 1))
            logits = model.lm_head(hidden.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)
            next_id = torch.argmax(logits, dim=-1).to(torch.int32)
            tokens.append(next_id.item())
            per_token_times.append(time.time() - t0)
        t_decode_end = time.time()

    return {
        'tokens': tokens,
        'text': tokenizer.decode(tokens),
        'prefill_ms': (t_prefill_end - t_prefill_start) * 1000,
        'decode_ms': (t_decode_end - t_decode_start) * 1000,
        'per_token_ms': per_token_times,
    }


def run_coreml_decode(out_dir, label, tokenizer, input_ids, max_gen, compute_unit):
    """Run CoreML greedy decode with stateless linear attention. Returns tokens + timing."""
    prompt_len = input_ids.shape[1]

    # Load models
    embed = ct.models.MLModel(os.path.join(out_dir, "embeddings.mlpackage"), compute_units=compute_unit)
    lmhead = ct.models.MLModel(os.path.join(out_dir, "lm_head.mlpackage"), compute_units=compute_unit)
    ffns = []
    states = []
    lin_convs = []
    lin_recs = []
    for ci in range(NUM_CHUNKS):
        path = os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage")
        m = ct.models.MLModel(path, compute_units=compute_unit)
        ffns.append(m)
        states.append(m.make_state())
        # Probe for linear state shapes
        spec = m.get_spec()
        inp_map = {inp.name: tuple(inp.type.multiArrayType.shape)
                   for inp in spec.description.input}
        if 'linear_conv_state' in inp_map:
            lin_convs.append(np.zeros(inp_map['linear_conv_state'], dtype=np.float16))
            lin_recs.append(np.zeros(inp_map['linear_recurrent_state'], dtype=np.float16))
        else:
            lin_convs.append(None)
            lin_recs.append(None)

    tokens = []

    # Prefill
    t_prefill_start = time.time()
    for pos in range(prompt_len):
        tok = input_ids[:, pos:pos+1].numpy().astype(np.int32)
        embed_out = embed.predict({"input_ids": tok})
        hidden = list(embed_out.values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos+1] = 0
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
            }
            if lin_convs[ci] is not None:
                inp["linear_conv_state"] = lin_convs[ci]
                inp["linear_recurrent_state"] = lin_recs[ci]
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']

    lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    if "argmax_idx" in lm_out:
        next_id = int(lm_out["argmax_idx"].flatten()[0])
    else:
        next_id = int(np.argmax(lm_out["logits"].flatten()))
    tokens.append(next_id)
    t_prefill_end = time.time()

    # Decode
    t_decode_start = time.time()
    per_token_times = []
    for gi in range(max_gen - 1):
        t0 = time.time()
        pos = prompt_len + gi
        tok = np.array([[next_id]], dtype=np.int32)
        embed_out = embed.predict({"input_ids": tok})
        hidden = list(embed_out.values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos+1] = 0
        for ci in range(NUM_CHUNKS):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
            }
            if lin_convs[ci] is not None:
                inp["linear_conv_state"] = lin_convs[ci]
                inp["linear_recurrent_state"] = lin_recs[ci]
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lin_convs[ci] = out['linear_conv_state_out']
                lin_recs[ci] = out['linear_recurrent_state_out']

        lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "argmax_idx" in lm_out:
            next_id = int(lm_out["argmax_idx"].flatten()[0])
        else:
            next_id = int(np.argmax(lm_out["logits"].flatten()))
        tokens.append(next_id)
        per_token_times.append(time.time() - t0)
    t_decode_end = time.time()

    # Cleanup
    del embed, lmhead
    for m in ffns:
        del m
    gc.collect()

    return {
        'tokens': tokens,
        'text': tokenizer.decode(tokens),
        'prefill_ms': (t_prefill_end - t_prefill_start) * 1000,
        'decode_ms': (t_decode_end - t_decode_start) * 1000,
        'per_token_ms': per_token_times,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--prompt", type=str,
                        default="Explain the difference between a stack and a queue in computer science.")
    parser.add_argument("--export-dir", type=str, default="/tmp/lut_vs_nolut_export")
    parser.add_argument("--skip-export", action="store_true", help="Skip export, use existing models")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch reference")
    args = parser.parse_args()
    max_gen = args.tokens
    out_dir = args.export_dir
    os.makedirs(out_dir, exist_ok=True)
    compute_unit = ct.ComputeUnit.CPU_AND_NE

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    input_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True).input_ids
    prompt_len = input_ids.shape[1]
    if prompt_len >= CTX - max_gen:
        input_ids = input_ids[:, :CTX - max_gen]
        prompt_len = input_ids.shape[1]

    print("=" * 70)
    print("  LUT4 vs No-LUT Text Generation Comparison")
    print(f"  Prompt ({prompt_len} tokens): {args.prompt[:80]}")
    print(f"  Generating {max_gen} tokens, CTX={CTX}")
    print("=" * 70)

    # ── Load & Export ──
    if not args.skip_export:
        cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
        cfg.context_length = CTX; cfg.state_length = CTX
        model = Qwen35ForCausalLM(cfg)
        assert model.load_pretrained_weights(MODEL_PATH)
        model.eval()
        for p in model.parameters(): p.requires_grad = False

        export_all_chunks(model, cfg, out_dir, lut_bits=4, per_channel=8)
        export_all_chunks(model, cfg, out_dir, lut_bits=None)
    else:
        cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
        cfg.context_length = CTX; cfg.state_length = CTX
        model = None

    results = {}

    # ── PyTorch Reference ──
    if not args.skip_pytorch:
        print("\n" + "=" * 60)
        print("  PyTorch Reference (greedy decode)")
        print("=" * 60)
        if model is None:
            model = Qwen35ForCausalLM(cfg)
            assert model.load_pretrained_weights(MODEL_PATH)
            model.eval()
            for p in model.parameters(): p.requires_grad = False
        results['pytorch'] = run_pytorch_decode(model, cfg, tokenizer, input_ids, max_gen)
        print(f"  Prefill: {results['pytorch']['prefill_ms']:.1f}ms")
        print(f"  Decode:  {results['pytorch']['decode_ms']:.1f}ms")
        print(f"  Text: {results['pytorch']['text'][:200]}")

    # Free model
    if model is not None:
        del model; gc.collect()

    # ── CoreML LUT4 ──
    print("\n" + "=" * 60)
    print("  CoreML Stateless + LUT4 (greedy decode, ANE)")
    print("=" * 60)
    results['lut4'] = run_coreml_decode(out_dir, "LUT4", tokenizer, input_ids, max_gen, compute_unit)
    print(f"  Prefill: {results['lut4']['prefill_ms']:.1f}ms")
    print(f"  Decode:  {results['lut4']['decode_ms']:.1f}ms")
    print(f"  Text: {results['lut4']['text'][:200]}")

    # ── CoreML no-LUT ──
    print("\n" + "=" * 60)
    print("  CoreML Stateless + No LUT / fp16 (greedy decode, ANE)")
    print("=" * 60)
    results['fp16'] = run_coreml_decode(out_dir, "fp16", tokenizer, input_ids, max_gen, compute_unit)
    print(f"  Prefill: {results['fp16']['prefill_ms']:.1f}ms")
    print(f"  Decode:  {results['fp16']['decode_ms']:.1f}ms")
    print(f"  Text: {results['fp16']['text'][:200]}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    configs = []
    if 'pytorch' in results:
        configs.append(('PyTorch', results['pytorch']))
    configs.append(('Stateless+LUT4', results['lut4']))
    configs.append(('Stateless+fp16', results['fp16']))

    # Latency table
    print(f"\n{'Config':<20} {'Prefill(ms)':>12} {'Decode(ms)':>12} {'ms/token':>10} {'tok/s':>8}")
    print("-" * 65)
    for name, r in configs:
        avg_ms = np.mean(r['per_token_ms']) * 1000 if r['per_token_ms'] else 0
        tps = 1000.0 / avg_ms if avg_ms > 0 else 0
        print(f"  {name:<18} {r['prefill_ms']:>12.1f} {r['decode_ms']:>12.1f} {avg_ms:>10.1f} {tps:>8.1f}")

    # Token match
    if 'pytorch' in results:
        pt_toks = results['pytorch']['tokens']
        for name, r in configs:
            if name == 'PyTorch': continue
            matches = sum(1 for a, b in zip(pt_toks, r['tokens']) if a == b)
            total = min(len(pt_toks), len(r['tokens']))
            print(f"\n  {name} vs PyTorch: {matches}/{total} tokens match ({100*matches/total:.1f}%)")

    # LUT4 vs fp16 direct
    l4_toks = results['lut4']['tokens']
    f16_toks = results['fp16']['tokens']
    matches = sum(1 for a, b in zip(l4_toks, f16_toks) if a == b)
    total = min(len(l4_toks), len(f16_toks))
    print(f"\n  LUT4 vs fp16 direct: {matches}/{total} tokens match ({100*matches/total:.1f}%)")

    # Generated text
    print(f"\n{'='*70}")
    print("  GENERATED TEXT")
    print(f"{'='*70}")
    for name, r in configs:
        print(f"\n--- {name} ---")
        print(r['text'][:500])

    # Per-token detail (first 20)
    print(f"\n{'='*70}")
    print("  TOKEN DETAIL (first 20)")
    print(f"{'='*70}")
    header_parts = [f"{'Pos':>4}"]
    for name, _ in configs:
        header_parts.append(f"{name:>20}")
    print(" ".join(header_parts))
    print("-" * (4 + 21 * len(configs)))
    for i in range(min(20, max_gen)):
        parts = [f"  {i:>2}"]
        for name, r in configs:
            tok_id = r['tokens'][i] if i < len(r['tokens']) else -1
            tok_str = tokenizer.decode([tok_id])[:12] if tok_id >= 0 else "?"
            parts.append(f"{tok_id:>6} {tok_str:<13}")
        print(" ".join(parts))

    print(f"\nExport dir: {out_dir}")


if __name__ == "__main__":
    main()
