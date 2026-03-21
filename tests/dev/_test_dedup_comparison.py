#!/usr/bin/env python3
"""Compare Stateless models WITH vs WITHOUT ANEMLL-Dedup.

Exports all 4 FFN chunks as decode + prefill pairs, then:
  A) Keeps them separate (no dedup) — measures combined size
  B) Combines with ANEMLL-Dedup — measures combined size
  C) Runs end-to-end greedy decode through both and compares:
     - Model size (disk)
     - Latency (prefill + decode)
     - Generated text quality (token match vs PyTorch reference)

Usage:
    python tests/dev/_test_dedup_comparison.py --tokens 50
    python tests/dev/_test_dedup_comparison.py --tokens 50 --skip-export  # reuse existing
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
from anemll.utils.combine_models import _save_multifunction_dedup
from transformers import AutoTokenizer

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
CTX = 256
NUM_CHUNKS = 4
CHUNKS = [(0, 8), (8, 16), (16, 24), (24, 32)]
LAYERS_PER_CHUNK = 8


def dir_size_mb(path):
    """Get directory size in MB."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def export_decode_prefill_chunks(model, cfg, out_dir, lut_bits, per_channel=8):
    """Export decode + prefill for all 4 chunks."""
    label = f"LUT{lut_bits}" if lut_bits else "fp16"
    print(f"\n{'='*60}")
    print(f"  Exporting {label} decode + prefill ({NUM_CHUNKS} chunks)")
    print(f"{'='*60}")

    # Shared: embeddings
    embed_path = os.path.join(out_dir, "embeddings.mlpackage")
    if not os.path.exists(embed_path):
        print("  Exporting embeddings...")
        conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                               num_chunks=NUM_CHUNKS, lut_bits=None)
        ml = conv.convert_part_1(model)
        ml.save(embed_path)
        del ml, conv; gc.collect()

    # Shared: lm_head
    lmhead_path = os.path.join(out_dir, "lm_head.mlpackage")
    if not os.path.exists(lmhead_path):
        print("  Exporting lm_head...")
        conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                               num_chunks=NUM_CHUNKS, lut_bits=None)
        ml = conv.convert_part_3(model, argmax_in_model=False)
        ml.save(lmhead_path)
        del ml, conv; gc.collect()

    for ci in range(NUM_CHUNKS):
        # Decode chunk
        dec_path = os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage")
        if not os.path.exists(dec_path):
            t0 = time.time()
            print(f"  Exporting decode chunk {ci} ({label})...")
            conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                                   num_chunks=NUM_CHUNKS, lut_bits=lut_bits,
                                   per_channel=per_channel)
            ml = conv.convert_part_2(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
            ml.save(dec_path)
            del ml, conv; gc.collect()
            print(f"    Done ({time.time()-t0:.1f}s)")
        else:
            print(f"  Decode chunk {ci} exists, skipping.")

        # Prefill chunk
        pf_path = os.path.join(out_dir, f"prefill_{label}_chunk{ci}.mlpackage")
        if not os.path.exists(pf_path):
            t0 = time.time()
            print(f"  Exporting prefill chunk {ci} ({label})...")
            conv = Qwen35Converter(model, context_length=CTX, batch_size=1,
                                   num_chunks=NUM_CHUNKS, lut_bits=lut_bits,
                                   per_channel=per_channel)
            ml = conv.convert_part_2_prefill(model, chunk_idx=ci, total_chunks=NUM_CHUNKS)
            ml.save(pf_path)
            del ml, conv; gc.collect()
            print(f"    Done ({time.time()-t0:.1f}s)")
        else:
            print(f"  Prefill chunk {ci} exists, skipping.")


def combine_with_dedup(out_dir, label, use_dedup=True):
    """Combine decode+prefill per chunk with optional dedup. Returns combined dir."""
    tag = "dedup" if use_dedup else "nodedup"
    combined_dir = os.path.join(out_dir, f"combined_{label}_{tag}")
    os.makedirs(combined_dir, exist_ok=True)

    total_size_mb = 0.0
    for ci in range(NUM_CHUNKS):
        combined_path = os.path.join(combined_dir, f"chunk{ci}.mlpackage")
        if os.path.exists(combined_path):
            total_size_mb += dir_size_mb(combined_path)
            print(f"  Chunk {ci} combined ({tag}) already exists, skipping.")
            continue

        dec_path = os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage")
        pf_path = os.path.join(out_dir, f"prefill_{label}_chunk{ci}.mlpackage")
        sources = [
            (dec_path, "main", "infer"),
            (pf_path, "main", "prefill"),
        ]
        t0 = time.time()
        print(f"  Combining chunk {ci} ({tag})...")
        _save_multifunction_dedup(sources, combined_path,
                                  dedup_weights=use_dedup, verbose=False)
        dt = time.time() - t0
        sz = dir_size_mb(combined_path)
        total_size_mb += sz
        print(f"    Done ({dt:.1f}s) — {sz:.1f} MB")

    return combined_dir, total_size_mb


def run_pytorch_decode(model, cfg, tokenizer, input_ids, max_gen):
    """PyTorch greedy decode (reference)."""
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
        t_pf = time.time()
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
        t_pf_end = time.time()

        t_dec = time.time()
        per_tok = []
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
            per_tok.append(time.time() - t0)
        t_dec_end = time.time()

    return {
        'tokens': tokens, 'text': tokenizer.decode(tokens),
        'prefill_ms': (t_pf_end - t_pf) * 1000, 'decode_ms': (t_dec_end - t_dec) * 1000,
        'per_token_ms': per_tok,
    }


def run_coreml_decode_separate(out_dir, label, tokenizer, input_ids, max_gen, compute_unit):
    """Run decode using SEPARATE decode-only .mlpackage files (no dedup, no prefill model used)."""
    prompt_len = input_ids.shape[1]
    embed = ct.models.MLModel(os.path.join(out_dir, "embeddings.mlpackage"), compute_units=compute_unit)
    lmhead = ct.models.MLModel(os.path.join(out_dir, "lm_head.mlpackage"), compute_units=compute_unit)
    ffns, states, lcs, lrs = [], [], [], []
    for ci in range(NUM_CHUNKS):
        m = ct.models.MLModel(os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage"), compute_units=compute_unit)
        ffns.append(m); states.append(m.make_state())
        spec = m.get_spec()
        inp_map = {inp.name: tuple(inp.type.multiArrayType.shape) for inp in spec.description.input}
        lcs.append(np.zeros(inp_map['linear_conv_state'], dtype=np.float16) if 'linear_conv_state' in inp_map else None)
        lrs.append(np.zeros(inp_map['linear_recurrent_state'], dtype=np.float16) if 'linear_recurrent_state' in inp_map else None)

    tokens = []
    t_pf = time.time()
    for pos in range(prompt_len):
        tok = input_ids[:, pos:pos+1].numpy().astype(np.int32)
        hidden = list(embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16); mask[:, :, :, :pos+1] = 0
        for ci in range(NUM_CHUNKS):
            inp = {"hidden_states": hidden.astype(np.float16), "position_ids": np.array([pos], dtype=np.int32),
                   "causal_mask": mask, "current_pos": np.array([pos], dtype=np.int32)}
            if lcs[ci] is not None:
                inp["linear_conv_state"] = lcs[ci]; inp["linear_recurrent_state"] = lrs[ci]
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lcs[ci] = out['linear_conv_state_out']; lrs[ci] = out['linear_recurrent_state_out']
    lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    next_id = int(np.argmax(lm_out["logits"].flatten())) if "logits" in lm_out else int(lm_out["argmax_idx"].flatten()[0])
    tokens.append(next_id)
    t_pf_end = time.time()

    t_dec = time.time()
    per_tok = []
    for gi in range(max_gen - 1):
        t0 = time.time()
        pos = prompt_len + gi
        hidden = list(embed.predict({"input_ids": np.array([[next_id]], dtype=np.int32)}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16); mask[:, :, :, :pos+1] = 0
        for ci in range(NUM_CHUNKS):
            inp = {"hidden_states": hidden.astype(np.float16), "position_ids": np.array([pos], dtype=np.int32),
                   "causal_mask": mask, "current_pos": np.array([pos], dtype=np.int32)}
            if lcs[ci] is not None:
                inp["linear_conv_state"] = lcs[ci]; inp["linear_recurrent_state"] = lrs[ci]
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lcs[ci] = out['linear_conv_state_out']; lrs[ci] = out['linear_recurrent_state_out']
        lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        next_id = int(np.argmax(lm_out["logits"].flatten())) if "logits" in lm_out else int(lm_out["argmax_idx"].flatten()[0])
        tokens.append(next_id)
        per_tok.append(time.time() - t0)
    t_dec_end = time.time()
    del embed, lmhead
    for m in ffns: del m
    gc.collect()
    return {
        'tokens': tokens, 'text': tokenizer.decode(tokens),
        'prefill_ms': (t_pf_end - t_pf) * 1000, 'decode_ms': (t_dec_end - t_dec) * 1000,
        'per_token_ms': per_tok,
    }


def run_coreml_decode_combined(combined_dir, out_dir, tokenizer, input_ids, max_gen, compute_unit):
    """Run decode using COMBINED multifunction .mlpackage (dedup or no-dedup).
    Uses 'infer' function for single-token decode (same as separate decode model)."""
    prompt_len = input_ids.shape[1]
    embed = ct.models.MLModel(os.path.join(out_dir, "embeddings.mlpackage"), compute_units=compute_unit)
    lmhead = ct.models.MLModel(os.path.join(out_dir, "lm_head.mlpackage"), compute_units=compute_unit)
    ffns, states, lcs, lrs = [], [], [], []
    for ci in range(NUM_CHUNKS):
        m = ct.models.MLModel(os.path.join(combined_dir, f"chunk{ci}.mlpackage"),
                              compute_units=compute_unit, function_name="infer")
        ffns.append(m); states.append(m.make_state())
        spec = m.get_spec()
        # Find inputs for 'infer' function
        inp_map = {}
        for inp in spec.description.input:
            try:
                shape = tuple(inp.type.multiArrayType.shape)
                inp_map[inp.name] = shape
            except Exception:
                pass
        lcs.append(np.zeros(inp_map['linear_conv_state'], dtype=np.float16) if 'linear_conv_state' in inp_map else None)
        lrs.append(np.zeros(inp_map['linear_recurrent_state'], dtype=np.float16) if 'linear_recurrent_state' in inp_map else None)

    tokens = []
    t_pf = time.time()
    for pos in range(prompt_len):
        tok = input_ids[:, pos:pos+1].numpy().astype(np.int32)
        hidden = list(embed.predict({"input_ids": tok}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16); mask[:, :, :, :pos+1] = 0
        for ci in range(NUM_CHUNKS):
            inp = {"hidden_states": hidden.astype(np.float16), "position_ids": np.array([pos], dtype=np.int32),
                   "causal_mask": mask, "current_pos": np.array([pos], dtype=np.int32)}
            if lcs[ci] is not None:
                inp["linear_conv_state"] = lcs[ci]; inp["linear_recurrent_state"] = lrs[ci]
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lcs[ci] = out['linear_conv_state_out']; lrs[ci] = out['linear_recurrent_state_out']
    lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
    next_id = int(np.argmax(lm_out["logits"].flatten())) if "logits" in lm_out else int(lm_out["argmax_idx"].flatten()[0])
    tokens.append(next_id)
    t_pf_end = time.time()

    t_dec = time.time()
    per_tok = []
    for gi in range(max_gen - 1):
        t0 = time.time()
        pos = prompt_len + gi
        hidden = list(embed.predict({"input_ids": np.array([[next_id]], dtype=np.int32)}).values())[0]
        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16); mask[:, :, :, :pos+1] = 0
        for ci in range(NUM_CHUNKS):
            inp = {"hidden_states": hidden.astype(np.float16), "position_ids": np.array([pos], dtype=np.int32),
                   "causal_mask": mask, "current_pos": np.array([pos], dtype=np.int32)}
            if lcs[ci] is not None:
                inp["linear_conv_state"] = lcs[ci]; inp["linear_recurrent_state"] = lrs[ci]
            out = ffns[ci].predict(inp, state=states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                lcs[ci] = out['linear_conv_state_out']; lrs[ci] = out['linear_recurrent_state_out']
        lm_out = lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        next_id = int(np.argmax(lm_out["logits"].flatten())) if "logits" in lm_out else int(lm_out["argmax_idx"].flatten()[0])
        tokens.append(next_id)
        per_tok.append(time.time() - t0)
    t_dec_end = time.time()
    del embed, lmhead
    for m in ffns: del m
    gc.collect()
    return {
        'tokens': tokens, 'text': tokenizer.decode(tokens),
        'prefill_ms': (t_pf_end - t_pf) * 1000, 'decode_ms': (t_dec_end - t_dec) * 1000,
        'per_token_ms': per_tok,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--prompt", type=str,
                        default="Explain the difference between a stack and a queue in computer science.")
    parser.add_argument("--export-dir", type=str, default="/tmp/dedup_comparison")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--lut", type=int, default=4, help="LUT bits (4 or None for fp16)")
    args = parser.parse_args()
    max_gen = args.tokens
    out_dir = args.export_dir
    os.makedirs(out_dir, exist_ok=True)
    compute_unit = ct.ComputeUnit.CPU_AND_NE
    lut_bits = args.lut
    label = f"LUT{lut_bits}" if lut_bits else "fp16"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    input_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True).input_ids
    prompt_len = input_ids.shape[1]
    if prompt_len >= CTX - max_gen:
        input_ids = input_ids[:, :CTX - max_gen]
        prompt_len = input_ids.shape[1]

    print("=" * 70)
    print("  ANEMLL-Dedup Comparison: Combined vs Separate Models")
    print(f"  Config: Stateless {label}, {NUM_CHUNKS} chunks, CTX={CTX}")
    print(f"  Prompt ({prompt_len} tokens): {args.prompt[:80]}")
    print(f"  Generating {max_gen} tokens")
    print("=" * 70)

    # ── 1. Export ──
    cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
    cfg.context_length = CTX; cfg.state_length = CTX

    if not args.skip_export:
        model = Qwen35ForCausalLM(cfg)
        assert model.load_pretrained_weights(MODEL_PATH)
        model.eval()
        for p in model.parameters(): p.requires_grad = False
        export_decode_prefill_chunks(model, cfg, out_dir, lut_bits=lut_bits)
    else:
        model = None

    # ── 2. Measure separate model sizes ──
    print(f"\n{'='*60}")
    print("  Model Sizes (Separate)")
    print(f"{'='*60}")
    sep_total = 0.0
    for ci in range(NUM_CHUNKS):
        dec_sz = dir_size_mb(os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage"))
        pf_sz = dir_size_mb(os.path.join(out_dir, f"prefill_{label}_chunk{ci}.mlpackage"))
        print(f"  Chunk {ci}: decode={dec_sz:.1f}MB  prefill={pf_sz:.1f}MB  total={dec_sz+pf_sz:.1f}MB")
        sep_total += dec_sz + pf_sz
    embed_sz = dir_size_mb(os.path.join(out_dir, "embeddings.mlpackage"))
    lmhead_sz = dir_size_mb(os.path.join(out_dir, "lm_head.mlpackage"))
    sep_total += embed_sz + lmhead_sz
    print(f"  Embeddings: {embed_sz:.1f}MB  LM Head: {lmhead_sz:.1f}MB")
    print(f"  TOTAL (separate): {sep_total:.1f}MB")

    # ── 3. Combine WITHOUT dedup ──
    print(f"\n{'='*60}")
    print("  Combining WITHOUT dedup")
    print(f"{'='*60}")
    nodedup_dir, nodedup_ffn_total = combine_with_dedup(out_dir, label, use_dedup=False)
    nodedup_total = nodedup_ffn_total + embed_sz + lmhead_sz
    print(f"  Combined FFN total: {nodedup_ffn_total:.1f}MB")
    print(f"  TOTAL (combined, no dedup): {nodedup_total:.1f}MB")

    # ── 4. Combine WITH dedup ──
    print(f"\n{'='*60}")
    print("  Combining WITH ANEMLL-Dedup")
    print(f"{'='*60}")
    dedup_dir, dedup_ffn_total = combine_with_dedup(out_dir, label, use_dedup=True)
    dedup_total = dedup_ffn_total + embed_sz + lmhead_sz
    print(f"  Combined FFN total: {dedup_ffn_total:.1f}MB")
    print(f"  TOTAL (combined, dedup): {dedup_total:.1f}MB")

    # ── 5. Size comparison ──
    print(f"\n{'='*60}")
    print("  SIZE COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Config':<35} {'Size (MB)':>10} {'Saving':>10}")
    print(f"  {'-'*57}")
    print(f"  {'Separate (decode+prefill)':<35} {sep_total:>10.1f} {'---':>10}")
    print(f"  {'Combined (no dedup)':<35} {nodedup_total:>10.1f} {(1-nodedup_total/sep_total)*100:>9.1f}%")
    print(f"  {'Combined (ANEMLL-Dedup)':<35} {dedup_total:>10.1f} {(1-dedup_total/sep_total)*100:>9.1f}%")

    results = {}

    # ── 6. PyTorch reference ──
    if not args.skip_pytorch:
        print(f"\n{'='*60}")
        print("  PyTorch Reference")
        print(f"{'='*60}")
        if model is None:
            model = Qwen35ForCausalLM(cfg)
            assert model.load_pretrained_weights(MODEL_PATH)
            model.eval()
            for p in model.parameters(): p.requires_grad = False
        results['PyTorch'] = run_pytorch_decode(model, cfg, tokenizer, input_ids, max_gen)
        print(f"  Done. Text: {results['PyTorch']['text'][:120]}")
    if model is not None:
        del model; gc.collect()

    # ── 7. CoreML separate ──
    print(f"\n{'='*60}")
    print(f"  CoreML Separate (no prefill model used)")
    print(f"{'='*60}")
    results['Separate'] = run_coreml_decode_separate(out_dir, label, tokenizer, input_ids, max_gen, compute_unit)
    print(f"  Done. Text: {results['Separate']['text'][:120]}")

    # ── 8. CoreML combined no-dedup ──
    print(f"\n{'='*60}")
    print(f"  CoreML Combined (no dedup)")
    print(f"{'='*60}")
    results['NoDedup'] = run_coreml_decode_combined(nodedup_dir, out_dir, tokenizer, input_ids, max_gen, compute_unit)
    print(f"  Done. Text: {results['NoDedup']['text'][:120]}")

    # ── 9. CoreML combined with-dedup ──
    print(f"\n{'='*60}")
    print(f"  CoreML Combined (ANEMLL-Dedup)")
    print(f"{'='*60}")
    results['Dedup'] = run_coreml_decode_combined(dedup_dir, out_dir, tokenizer, input_ids, max_gen, compute_unit)
    print(f"  Done. Text: {results['Dedup']['text'][:120]}")

    # ── 10. Summary ──
    print(f"\n{'='*70}")
    print("  FULL COMPARISON SUMMARY")
    print(f"{'='*70}")

    configs = list(results.items())

    # Latency
    print(f"\n{'Config':<25} {'Prefill(ms)':>12} {'Decode(ms)':>12} {'ms/tok':>10} {'tok/s':>8}")
    print("-" * 70)
    for name, r in configs:
        avg_ms = np.mean(r['per_token_ms']) * 1000 if r['per_token_ms'] else 0
        tps = 1000.0 / avg_ms if avg_ms > 0 else 0
        print(f"  {name:<23} {r['prefill_ms']:>12.1f} {r['decode_ms']:>12.1f} {avg_ms:>10.1f} {tps:>8.1f}")

    # Token match vs PyTorch
    if 'PyTorch' in results:
        pt_toks = results['PyTorch']['tokens']
        print(f"\n  Token match vs PyTorch:")
        for name, r in configs:
            if name == 'PyTorch': continue
            matches = sum(1 for a, b in zip(pt_toks, r['tokens']) if a == b)
            total = min(len(pt_toks), len(r['tokens']))
            print(f"    {name:<23} {matches}/{total} ({100*matches/total:.1f}%)")

    # Check if dedup changes tokens vs separate
    sep_toks = results['Separate']['tokens']
    for name in ['NoDedup', 'Dedup']:
        if name in results:
            r = results[name]
            matches = sum(1 for a, b in zip(sep_toks, r['tokens']) if a == b)
            total = min(len(sep_toks), len(r['tokens']))
            print(f"\n  {name} vs Separate: {matches}/{total} tokens match ({100*matches/total:.1f}%)")

    # Model size
    print(f"\n  Model size comparison:")
    print(f"    Separate decode+prefill:  {sep_total:.1f} MB")
    print(f"    Combined (no dedup):      {nodedup_total:.1f} MB  ({(1-nodedup_total/sep_total)*100:.1f}% saving)")
    print(f"    Combined (ANEMLL-Dedup):  {dedup_total:.1f} MB  ({(1-dedup_total/sep_total)*100:.1f}% saving)")
    if dedup_ffn_total < nodedup_ffn_total:
        print(f"    Dedup FFN saving vs no-dedup: {nodedup_ffn_total - dedup_ffn_total:.1f} MB ({(1-dedup_ffn_total/nodedup_ffn_total)*100:.1f}%)")

    # Generated text
    print(f"\n{'='*70}")
    print("  GENERATED TEXT")
    print(f"{'='*70}")
    for name, r in configs:
        print(f"\n  --- {name} ---")
        print(f"  {r['text'][:300]}")

    print(f"\nExport dir: {out_dir}")


if __name__ == "__main__":
    main()
