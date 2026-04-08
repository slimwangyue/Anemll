#!/usr/bin/env python3
"""Qwen3.5-4B — Step 1: Export all CoreML model components.

Exports (with --nosplit-lmhead --lut-bits 4):
  - embeddings (LUT6 gs=8)                  → embeddings.mlpackage
  - lm_head_nosplit (LUT6 gs=8, single out)  → lm_head_nosplit.mlpackage
  - 9 FFN decode chunks (LUT4 gs=4)          → ffn_LUT4_chunk{0..8}.mlpackage
  - 9 FFN prefill chunks (LUT4 gs=4)         → prefill_LUT4_chunk{0..8}.mlpackage

Usage:
    python scripts_qwen3_5/export.py --model /path/to/Qwen3.5-4B --output /path/to/output
    python scripts_qwen3_5/export.py --nosplit-lmhead --lut-bits 4 --per-channel 4
    python scripts_qwen3_5/export.py --skip-existing
"""
import gc, time, argparse, os, sys, shutil, glob
import numpy as np
import torch
import coremltools as ct

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)  # must be first for config.py

from config import (
    BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS, LM_HEAD_LUT,
    PER_CHANNEL, FFN_PER_CHANNEL, FFN_LABEL, DEFAULT_HF_MODEL, DEFAULT_OUTPUT,
    CHUNK_RANGES,
)
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter


def export_embeddings(model, out_dir, skip_existing, compute_precision="float16"):
    path = os.path.join(out_dir, "embeddings.mlpackage")
    if skip_existing and os.path.exists(path):
        print(f"  [skip] embeddings")
        return
    print(f"  Exporting embeddings (LUT{LUT_BITS} gs={PER_CHANNEL})...")
    t0 = time.time()
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=PER_CHANNEL,
                           compute_precision=compute_precision)
    ml = conv.convert_part_1(model)
    ml.save(path)
    del ml, conv; gc.collect()
    print(f"  Saved embeddings ({time.time()-t0:.1f}s)")


def export_lm_head(model, out_dir, skip_existing, compute_precision="float16"):
    path = os.path.join(out_dir, "lm_head_logits.mlpackage")
    if skip_existing and os.path.exists(path):
        print(f"  [skip] lm_head_logits")
        return
    print(f"  Exporting lm_head_logits 16-way split (LUT{LM_HEAD_LUT})...")
    t0 = time.time()
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LM_HEAD_LUT, per_channel=PER_CHANNEL,
                           compute_precision=compute_precision)
    ml = conv.convert_part_3(model, argmax_in_model=False)
    ml.save(path)
    del ml, conv; gc.collect()
    print(f"  Saved lm_head_logits ({time.time()-t0:.1f}s)")


def export_lm_head_nosplit(model, out_dir, skip_existing, compute_precision="float16"):
    """Export a NON-SPLIT lm_head (single Conv2d for full vocab)."""
    path = os.path.join(out_dir, "lm_head_nosplit.mlpackage")
    if skip_existing and os.path.exists(path):
        print(f"  [skip] lm_head_nosplit")
        return

    class LMHeadNoSplitWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            parts = [getattr(m, f"lm_head16_{i+1}").weight for i in range(m.lm_head_split)]
            full_weight = torch.cat(parts, dim=0)
            self.lm_head = torch.nn.Conv2d(
                m.config.hidden_size, m.config.vocab_size, 1, bias=False,
                dtype=MODEL_DTYPE,
            ).to(TEST_DEVICE)
            self.lm_head.weight.data.copy_(full_weight)

        def forward(self, hidden_states):
            h = hidden_states.permute(0, 2, 1).unsqueeze(2)
            logits = self.lm_head(h)
            logits = logits.squeeze(2).permute(0, 2, 1)
            return logits

    print(f"  Exporting lm_head_nosplit (LUT{LM_HEAD_LUT} gs={PER_CHANNEL})...")
    t0 = time.time()
    wrapper = LMHeadNoSplitWrapper(model).eval()
    sample_input = torch.zeros((1, 1, model.config.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample_input)

    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LM_HEAD_LUT, per_channel=PER_CHANNEL,
                           compute_precision=compute_precision)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_states", shape=sample_input.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        compute_precision=conv.compute_precision,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    if LM_HEAD_LUT:
        conv.converted_model = mlmodel
        conv.postprocess(num_workers=1)
        mlmodel = conv.converted_model

    mlmodel.save(path)
    del mlmodel, conv, wrapper; gc.collect()
    print(f"  Saved lm_head_nosplit ({time.time()-t0:.1f}s)")


def export_ffn_chunks(model, out_dir, skip_existing, only_chunk=None, static_prefill=False,
                      lut_bits_override=None, per_channel_override=None, compute_precision="float16"):
    lut_bits = lut_bits_override if lut_bits_override is not None else LUT_BITS
    ffn_pc = per_channel_override if per_channel_override is not None else FFN_PER_CHANNEL
    label = f"LUT{lut_bits}"
    chunk_indices = [only_chunk] if only_chunk is not None else list(range(NUM_CHUNKS))
    for ci in chunk_indices:
        sl, el = CHUNK_RANGES[ci]
        # Decode chunk
        dec_path = os.path.join(out_dir, f"ffn_{label}_chunk{ci}.mlpackage")
        if skip_existing and os.path.exists(dec_path):
            print(f"  [skip] decode chunk {ci} (layers {sl}-{el-1})")
        else:
            print(f"  Exporting decode chunk {ci} layers [{sl}-{el-1}] ({label} gs={ffn_pc})...")
            t0 = time.time()
            conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                                   num_chunks=NUM_CHUNKS, lut_bits=lut_bits, per_channel=ffn_pc,
                                   compute_precision=compute_precision)
            ml = conv.convert_part_2(model, chunk_idx=ci, total_chunks=NUM_CHUNKS,
                                     override_start_layer=sl, override_end_layer=el)
            ml.save(dec_path)
            del ml, conv; gc.collect()
            print(f"  Saved decode chunk {ci} ({time.time()-t0:.1f}s)")

        # Prefill chunk
        if static_prefill:
            pf_path = os.path.join(out_dir, f"prefill_{label}_chunk{ci}_bs{BATCH_SIZE}.mlpackage")
            pf_desc = f"prefill chunk {ci} layers [{sl}-{el-1}] static bs{BATCH_SIZE}"
        else:
            pf_path = os.path.join(out_dir, f"prefill_{label}_chunk{ci}.mlpackage")
            pf_desc = f"prefill chunk {ci} layers [{sl}-{el-1}]"
        if skip_existing and os.path.exists(pf_path):
            print(f"  [skip] {pf_desc}")
        else:
            print(f"  Exporting {pf_desc} ({label} gs={ffn_pc})...")
            t0 = time.time()
            conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                                   num_chunks=NUM_CHUNKS, lut_bits=lut_bits, per_channel=ffn_pc,
                                   compute_precision=compute_precision)
            if static_prefill:
                ml = conv.convert_part_2_prefill_exact(
                    model, chunk_idx=ci, total_chunks=NUM_CHUNKS,
                    exact_seq_len=BATCH_SIZE,
                    override_start_layer=sl, override_end_layer=el)
            else:
                ml = conv.convert_part_2_prefill(model, chunk_idx=ci, total_chunks=NUM_CHUNKS,
                                                 override_start_layer=sl, override_end_layer=el)
            ml.save(pf_path)
            del ml, conv; gc.collect()
            print(f"  Saved {pf_desc} ({time.time()-t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5-4B for ANE (Milestone 2.1)")
    parser.add_argument("--model", default=DEFAULT_HF_MODEL,
                        help="Path to HuggingFace Qwen3.5-4B model directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output directory for exported .mlpackage files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip export if .mlpackage already exists")
    parser.add_argument("--only-chunk", type=int, default=None,
                        help="Export only the specified chunk index")
    parser.add_argument("--chunks", type=str, default=None,
                        help="Comma-separated chunk indices to export, e.g. '0,1,2,3'")
    parser.add_argument("--ffn-only", action="store_true",
                        help="Only export FFN chunks (skip embeddings and lm_head)")
    parser.add_argument("--lut-bits", type=int, default=None,
                        help="Override FFN LUT bits (e.g. 4 for LUT4). Default: use config.py")
    parser.add_argument("--per-channel", type=int, default=None,
                        help="Override FFN per-channel group size. Default: use config.py")
    parser.add_argument("--static-prefill", action="store_true",
                        help="Use static-shape prefill (convert_part_2_prefill_exact) with valid_len")
    parser.add_argument("--fp32-compute", action="store_true",
                        help="Use FLOAT32 compute precision (default: FLOAT16)")
    parser.add_argument("--nosplit-lmhead", action="store_true",
                        help="Export lm_head as single Conv2d (no 16-way split). Required for embed_lmhead_combined.")
    args = parser.parse_args()

    # --chunks takes precedence over --only-chunk
    if args.chunks is not None:
        args._chunk_list = [int(x.strip()) for x in args.chunks.split(",")]
    elif args.only_chunk is not None:
        args._chunk_list = [args.only_chunk]
    else:
        args._chunk_list = None  # all chunks
    os.makedirs(args.output, exist_ok=True)

    prefill_mode = "static" if args.static_prefill else "dynamic"
    cp = "float32" if args.fp32_compute else "float16"
    print("=" * 70)
    print("  Qwen3.5-4B ANE Export — Milestone 2.1 (LUT6 gs=4 FFN)")
    print(f"  Embed: LUT{LUT_BITS} gs={PER_CHANNEL} | LM Head: LUT{LM_HEAD_LUT} gs={PER_CHANNEL} | FFN: {FFN_LABEL} gs={FFN_PER_CHANNEL} × {NUM_CHUNKS} chunks")
    print(f"  Chunk partition ([FLLL] 9-chunk): {CHUNK_RANGES}")
    print(f"  Batch: {BATCH_SIZE} | CTX: {CTX} | Prefill: {prefill_mode} | Compute: {cp.upper()}")
    if args._chunk_list is not None:
        print(f"  Chunks: {args._chunk_list}")
    if args.ffn_only:
        print(f"  Mode: FFN-only (skipping embeddings & lm_head)")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print("=" * 70)

    print("\nLoading model weights...")
    t_load = time.time()
    cfg = Qwen35Config.from_json(os.path.join(args.model, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(args.model), f"Failed to load weights from {args.model}"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t_load:.1f}s")

    t_total = time.time()
    print("\n[1/3] FFN Chunks (decode + prefill)")
    if args._chunk_list is not None:
        for ci in args._chunk_list:
            export_ffn_chunks(model, args.output, args.skip_existing,
                              only_chunk=ci, static_prefill=args.static_prefill,
                              lut_bits_override=args.lut_bits, per_channel_override=args.per_channel,
                              compute_precision=cp)
    else:
        export_ffn_chunks(model, args.output, args.skip_existing,
                          static_prefill=args.static_prefill,
                          lut_bits_override=args.lut_bits, per_channel_override=args.per_channel,
                          compute_precision=cp)
    if not args.ffn_only:
        print("\n[2/3] Embeddings")
        export_embeddings(model, args.output, args.skip_existing, compute_precision=cp)
        print("\n[3/3] LM Head")
        if args.nosplit_lmhead:
            export_lm_head_nosplit(model, args.output, args.skip_existing, compute_precision=cp)
        else:
            export_lm_head(model, args.output, args.skip_existing, compute_precision=cp)
    else:
        print("\n  [--ffn-only] Skipping embeddings and lm_head")
    
    del model; gc.collect()

    total_mb = 0
    print(f"\n{'='*70}")
    print("  EXPORT SUMMARY")
    print(f"{'='*70}")
    for f in sorted(os.listdir(args.output)):
        full = os.path.join(args.output, f)
        if os.path.isdir(full):
            sz = sum(os.path.getsize(os.path.join(dp, fn))
                     for dp, _, fns in os.walk(full) for fn in fns
                     if not os.path.islink(os.path.join(dp, fn))) / (1024 * 1024)
            total_mb += sz
            print(f"  {f:<50s} {sz:>8.1f} MB")
    print(f"  {'TOTAL':<50s} {total_mb:>8.1f} MB")
    # Copy tokenizer files so the output dir is self-contained
    tok_patterns = ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                    "merges.txt", "special_tokens_map.json"]
    copied = []
    for pat in tok_patterns:
        for src in glob.glob(os.path.join(args.model, pat)):
            dst = os.path.join(args.output, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied.append(os.path.basename(src))
    if copied:
        print(f"  Copied tokenizer files: {', '.join(copied)}")

    print(f"\n  Elapsed: {time.time()-t_total:.1f}s")
    print(f"\nNext: python scripts_qwen3_5/combine.py --input {args.output}")


if __name__ == "__main__":
    main()
