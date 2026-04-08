#!/usr/bin/env python3
"""Test non-split lm_head export + combined embed+lmhead multifunction model on ANE.

This script:
1. Exports a NON-SPLIT lm_head (single Conv2d instead of 16-way)
2. Exports embeddings as usual (reuses existing if available)
3. Combines both into a 2-function mlpackage with weight dedup
4. Tests both functions on ANE
5. Compares non-split lm_head output vs 16-way split lm_head

Usage:
    python tests/dev/test_lmhead_nosplit.py
"""
import gc
import os
import sys
import time
import shutil

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts_qwen3_5"))

import coremltools as ct
from config import (
    BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS, LM_HEAD_LUT, PER_CHANNEL,
    DEFAULT_HF_MODEL, DEFAULT_OUTPUT,
)
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter


# ── Directories ──────────────────────────────────────────────────────

HF_MODEL = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT, "lmhead_nosplit_test")
EXISTING_LUT6_DIR = DEFAULT_OUTPUT  # where the existing 16-split lm_head lives


def load_model():
    """Load the HF model."""
    print(f"Loading model from {HF_MODEL}...")
    t0 = time.time()
    config = Qwen35Config.from_json(os.path.join(HF_MODEL, "config.json"))
    config.context_length = CTX
    config.state_length = CTX
    model = Qwen35ForCausalLM(config)
    assert model.load_pretrained_weights(HF_MODEL), f"Failed to load weights from {HF_MODEL}"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model


def export_embeddings(model, out_dir):
    """Export embeddings model (or reuse from existing)."""
    path = os.path.join(out_dir, "embeddings.mlpackage")
    if os.path.exists(path):
        print(f"  [skip] embeddings already exists")
        return path

    # Try to reuse from existing LUT6 dir
    existing = os.path.join(EXISTING_LUT6_DIR, "embeddings.mlpackage")
    if os.path.exists(existing):
        print(f"  Copying embeddings from existing export...")
        shutil.copytree(existing, path)
        return path

    print(f"  Exporting embeddings (LUT{LUT_BITS} gs={PER_CHANNEL})...")
    t0 = time.time()
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=PER_CHANNEL,
    )
    ml = conv.convert_part_1(model)
    ml.save(path)
    del ml, conv
    gc.collect()
    print(f"  Saved embeddings ({time.time()-t0:.1f}s)")
    return path


def export_lmhead_nosplit(model, out_dir):
    """Export a NON-SPLIT lm_head (single Conv2d for full vocab)."""
    path = os.path.join(out_dir, "lm_head_nosplit.mlpackage")
    if os.path.exists(path):
        print(f"  [skip] lm_head_nosplit already exists")
        return path

    class LMHeadNoSplitWrapper(torch.nn.Module):
        """Single Conv2d lm_head — no 16-way split."""
        def __init__(self, model: Qwen35ForCausalLM) -> None:
            super().__init__()
            # Reconstruct the full lm_head weight from 16 shards
            parts = [getattr(model, f"lm_head16_{i+1}").weight for i in range(model.lm_head_split)]
            full_weight = torch.cat(parts, dim=0)  # [vocab, hidden, 1, 1]
            self.lm_head = torch.nn.Conv2d(
                model.config.hidden_size, model.config.vocab_size, 1, bias=False,
                dtype=MODEL_DTYPE,
            ).to(TEST_DEVICE)
            self.lm_head.weight.data.copy_(full_weight)

        def forward(self, hidden_states):
            # hidden_states: [1, seq_len, hidden_size]
            h = hidden_states.permute(0, 2, 1).unsqueeze(2)  # [1, hidden, 1, seq_len]
            logits = self.lm_head(h)  # [1, vocab, 1, seq_len]
            logits = logits.squeeze(2).permute(0, 2, 1)  # [1, seq_len, vocab]
            return logits

    print(f"  Exporting lm_head NO-SPLIT (LUT{LM_HEAD_LUT} gs={PER_CHANNEL})...")
    t0 = time.time()
    wrapper = LMHeadNoSplitWrapper(model).eval()
    sample_input = torch.zeros((1, 1, model.config.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample_input)

    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=LM_HEAD_LUT, per_channel=PER_CHANNEL,
    )
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_states", shape=sample_input.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        compute_precision=conv.compute_precision,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    # Apply LUT quantization
    if LM_HEAD_LUT:
        conv.converted_model = mlmodel
        conv.postprocess(num_workers=1)
        mlmodel = conv.converted_model

    mlmodel.save(path)
    sz = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(path)
        for f in fns
    ) / 1024 / 1024
    print(f"  Saved lm_head_nosplit ({time.time()-t0:.1f}s) — {sz:.1f} MB")
    del mlmodel, conv, wrapper
    gc.collect()
    return path


def combine_embed_lmhead(embed_path, lmhead_path, out_dir):
    """Combine embeddings + lm_head into a 2-function mlpackage with weight dedup."""
    combined_path = os.path.join(out_dir, "embed_lmhead_combined.mlpackage")
    if os.path.exists(combined_path):
        print(f"  [skip] combined model already exists")
        return combined_path

    print(f"  Combining embed + lm_head into multifunction model with dedup...")
    t0 = time.time()

    sources = [
        (embed_path, "main", "embed"),
        (lmhead_path, "main", "lmhead"),
    ]

    # Try with dedup first
    try:
        from anemll.utils.dedup_weights import prepare_dedup_sources, dedup_cross_model_blobs
        with prepare_dedup_sources(sources, verbose=True, preflight=False) as deduped:
            desc = ct.utils.MultiFunctionDescriptor()
            for path, src_fn, tgt_fn in deduped:
                desc.add_function(path, src_fn, tgt_fn)
            desc.default_function_name = "embed"
            ct.utils.save_multifunction(desc, combined_path)

        # Post-process: redirect identical weight blobs across functions
        print("  Running cross-model blob dedup post-processing...")
        saved = dedup_cross_model_blobs(combined_path, "embed", "lmhead", verbose=True)
        if saved > 0:
            print(f"  Cross-model dedup saved {saved / 1e6:.1f} MB")
        else:
            print("  Cross-model dedup: no savings (blobs may already be shared or differ)")
    except Exception as e:
        print(f"  Dedup failed ({e}), trying without dedup...")
        if os.path.exists(combined_path):
            shutil.rmtree(combined_path)
        desc = ct.utils.MultiFunctionDescriptor()
        for path, src_fn, tgt_fn in sources:
            desc.add_function(path, src_fn, tgt_fn)
        desc.default_function_name = "embed"
        ct.utils.save_multifunction(desc, combined_path)

    sz = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(combined_path)
        for f in fns
    ) / 1024 / 1024
    print(f"  Saved combined ({time.time()-t0:.1f}s) — {sz:.1f} MB")
    return combined_path


def test_on_ane(model_path, function_name=None, input_name="input_ids",
                input_data=None, label="test"):
    """Load a CoreML model and run on ANE."""
    print(f"\n  Testing {label} on ANE...")
    t0 = time.time()
    try:
        ml = ct.models.MLModel(
            model_path,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            function_name=function_name,
        )
        pred = ml.predict({input_name: input_data})
        elapsed = time.time() - t0
        print(f"  OK ({elapsed:.2f}s) — outputs: {list(pred.keys())}")
        for k, v in pred.items():
            if isinstance(v, np.ndarray):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}, "
                      f"min={v.min():.4f}, max={v.max():.4f}")
        del ml
        gc.collect()
        return pred
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def test_split_lmhead(lmhead_16split_path, hidden_input):
    """Test the existing 16-split lm_head for comparison."""
    print(f"\n  Testing 16-split lm_head for reference...")
    t0 = time.time()
    try:
        ml = ct.models.MLModel(
            lmhead_16split_path,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        pred = ml.predict({"hidden_states": hidden_input})
        elapsed = time.time() - t0
        # Reconstruct full logits from 16 parts
        parts = [pred[f"logits{i+1}"] for i in range(16)]
        logits = np.concatenate(parts, axis=-1)
        print(f"  OK ({elapsed:.2f}s) — reconstructed logits shape: {logits.shape}")
        del ml
        gc.collect()
        return logits
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model:  {HF_MODEL}")
    print(f"Config: CTX={CTX}, BATCH={BATCH_SIZE}, LUT_BITS={LUT_BITS}, "
          f"LM_HEAD_LUT={LM_HEAD_LUT}, PER_CHANNEL={PER_CHANNEL}")
    print()

    # ── Step 1: Load model ──
    model = load_model()
    print(f"  vocab_size={model.config.vocab_size}, hidden_size={model.config.hidden_size}")
    print()

    # ── Step 2: Export non-split lm_head ──
    print("=== Step 2: Export non-split lm_head ===")
    lmhead_path = export_lmhead_nosplit(model, OUTPUT_DIR)
    print()

    # ── Step 3: Export embeddings ──
    print("=== Step 3: Export embeddings ===")
    embed_path = export_embeddings(model, OUTPUT_DIR)
    print()

    # Free model memory
    del model
    gc.collect()

    # ── Step 4: Test non-split lm_head on ANE (standalone) ──
    print("=== Step 4: Test non-split lm_head on ANE ===")
    hidden_input = np.random.randn(1, 1, 2560).astype(np.float16)
    nosplit_pred = test_on_ane(
        lmhead_path,
        input_name="hidden_states",
        input_data=hidden_input,
        label="lm_head_nosplit (standalone)",
    )

    # ── Step 5: Test existing 16-split lm_head for comparison ──
    split16_path = os.path.join(EXISTING_LUT6_DIR, "lm_head_logits.mlpackage")
    split16_logits = None
    if os.path.exists(split16_path):
        print("=== Step 5: Test 16-split lm_head for comparison ===")
        split16_logits = test_split_lmhead(split16_path, hidden_input)

    # ── Step 6: Compare outputs ──
    if nosplit_pred is not None and split16_logits is not None:
        print("\n=== Step 6: Compare non-split vs 16-split ===")
        nosplit_logits = nosplit_pred["logits"]
        print(f"  nosplit shape: {nosplit_logits.shape}")
        print(f"  split16 shape: {split16_logits.shape}")

        # Cosine similarity
        a = nosplit_logits.flatten().astype(np.float32)
        b = split16_logits.flatten().astype(np.float32)
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        print(f"  Cosine similarity: {cos:.6f}")

        # Max abs diff
        diff = np.abs(a - b)
        print(f"  Max abs diff: {diff.max():.6f}")
        print(f"  Mean abs diff: {diff.mean():.6f}")

        # Top-1 agreement
        top1_nosplit = np.argmax(nosplit_logits, axis=-1)
        top1_split16 = np.argmax(split16_logits, axis=-1)
        agree = (top1_nosplit == top1_split16).mean()
        print(f"  Top-1 agreement: {agree*100:.1f}%")
        print(f"  nosplit top-1: {top1_nosplit.flatten()}")
        print(f"  split16 top-1: {top1_split16.flatten()}")

    # ── Step 7: Combine into multifunction model ──
    print("\n=== Step 7: Combine embed + lmhead ===")
    combined_path = combine_embed_lmhead(embed_path, lmhead_path, OUTPUT_DIR)

    # ── Step 8: Test combined model on ANE ──
    print("\n=== Step 8: Test combined model on ANE ===")
    # Test embed function
    token_input = np.array([[42]], dtype=np.int32)
    embed_pred = test_on_ane(
        combined_path,
        function_name="embed",
        input_name="input_ids",
        input_data=token_input,
        label="combined::embed",
    )

    # Test lmhead function
    lmhead_pred = test_on_ane(
        combined_path,
        function_name="lmhead",
        input_name="hidden_states",
        input_data=hidden_input,
        label="combined::lmhead",
    )

    # ── Step 9: Verify combined == standalone ──
    if lmhead_pred is not None and nosplit_pred is not None:
        print("\n=== Step 9: Verify combined == standalone ===")
        a = lmhead_pred["logits"].flatten().astype(np.float32)
        b = nosplit_pred["logits"].flatten().astype(np.float32)
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        print(f"  Combined vs standalone cosine: {cos:.6f}")

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for p in [lmhead_path, embed_path, combined_path]:
        if os.path.exists(p):
            sz = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(p)
                for f in fns
            ) / 1024 / 1024
            print(f"  {os.path.basename(p):40s} {sz:8.1f} MB")
    if os.path.exists(split16_path):
        sz = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(split16_path)
            for f in fns
        ) / 1024 / 1024
        print(f"  {'lm_head_logits.mlpackage (16-split)':40s} {sz:8.1f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
