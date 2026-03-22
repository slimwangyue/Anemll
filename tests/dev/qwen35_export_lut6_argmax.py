#!/usr/bin/env python3
"""Export Qwen3.5-4B LM head with LUT6 quantization + fused argmax.

Lightweight version: only loads lm_head weights (not full model) to avoid OOM
on 16GB machines.

Usage:
    python tests/dev/qwen35_export_lut6_argmax.py \
        --model /Users/yw68/local_llm/models/Qwen__Qwen3.5-4B \
        --output /Users/yw68/Anemll_remote_run/qwen35_milestone1_3
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import gc, time, shutil, argparse, json, warnings
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
import coremltools.optimize as cto
try:
    from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
except Exception:
    SklearnConvergenceWarning = None

# ── Config ──
BATCH_SIZE = 256
CTX = 1024
NUM_CHUNKS = 4
LM_HEAD_LUT = 6
PER_CHANNEL = 8
MODEL_DTYPE = torch.float16


class LMHeadWrapper(nn.Module):
    """Standalone LM head wrapper with optional argmax fusion."""
    def __init__(self, weight_tensor, argmax_mode=True):
        super().__init__()
        # Create Conv2d without default initialization to save memory
        vocab_size, hidden_size = weight_tensor.shape[0], weight_tensor.shape[1]
        self.lm_head = nn.Conv2d.__new__(nn.Conv2d)
        nn.Module.__init__(self.lm_head)
        self.lm_head.in_channels = hidden_size
        self.lm_head.out_channels = vocab_size
        self.lm_head.kernel_size = (1, 1)
        self.lm_head.stride = (1, 1)
        self.lm_head.padding = (0, 0)
        self.lm_head.dilation = (1, 1)
        self.lm_head.groups = 1
        self.lm_head.padding_mode = 'zeros'
        self.lm_head.transposed = False
        self.lm_head.output_padding = (0, 0)
        self.lm_head.weight = nn.Parameter(weight_tensor)
        self.lm_head.bias = None
        self.argmax_mode = argmax_mode

    def forward(self, hidden_states):
        logits = self.lm_head(hidden_states.permute(0, 2, 1).unsqueeze(2))
        logits = logits.squeeze(2).permute(0, 2, 1)
        if self.argmax_mode:
            argmax_idx_i64 = torch.argmax(logits, dim=-1)
            argmax_val = torch.gather(logits, -1, argmax_idx_i64.unsqueeze(-1)).squeeze(-1)
            argmax_idx = argmax_idx_i64.to(torch.int32)
            return argmax_idx, argmax_val
        return logits


def load_lm_head_weight(model_path, hidden_size, vocab_size):
    """Load only lm_head.weight from safetensors files (memory efficient)."""
    import safetensors.torch
    for fname in sorted(os.listdir(model_path)):
        if not fname.endswith(".safetensors"):
            continue
        full_path = os.path.join(model_path, fname)
        # Check if this file contains lm_head.weight
        with safetensors.torch.safe_open(full_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if "lm_head.weight" in keys:
                w = f.get_tensor("lm_head.weight")
                return w.view(vocab_size, hidden_size, 1, 1).to(MODEL_DTYPE)
    # Fallback: use embed_tokens.weight
    for fname in sorted(os.listdir(model_path)):
        if not fname.endswith(".safetensors"):
            continue
        full_path = os.path.join(model_path, fname)
        with safetensors.torch.safe_open(full_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for k in keys:
                if "embed_tokens.weight" in k:
                    w = f.get_tensor(k)
                    return w.view(vocab_size, hidden_size, 1, 1).to(MODEL_DTYPE)
    raise RuntimeError("Could not find lm_head.weight or embed_tokens.weight")


def palettize_model(mlmodel, lut_bits, per_channel):
    """Apply LUT quantization to a CoreML model."""
    with warnings.catch_warnings():
        if SklearnConvergenceWarning is not None:
            warnings.simplefilter("ignore", SklearnConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig
        cfg = OpPalettizerConfig(
            mode="kmeans",
            nbits=lut_bits,
            granularity="per_grouped_channel",
            group_size=per_channel,
            num_kmeans_workers=1,
        )
        config = OptimizationConfig(global_config=cfg)
        return cto.coreml.palettize_weights(mlmodel, config)


def main():
    parser = argparse.ArgumentParser(description="Export LUT6+argmax LM head for Qwen3.5-4B")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to HuggingFace Qwen3.5-4B model directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory (overwrites existing lm_head.mlpackage)")
    args = parser.parse_args()

    out_path = os.path.join(args.output, "lm_head.mlpackage")

    # Read config
    with open(os.path.join(args.model, "config.json")) as f:
        model_config = json.load(f)
    # Handle nested text_config for multimodal models
    text_cfg = model_config.get("text_config", model_config)
    hidden_size = text_cfg["hidden_size"]
    vocab_size = text_cfg["vocab_size"]

    print("=" * 70)
    print("  Qwen3.5-4B LM Head Export — LUT6 + Argmax (Lightweight)")
    print(f"  hidden_size: {hidden_size}, vocab_size: {vocab_size}")
    print(f"  LUT bits: {LM_HEAD_LUT}, per_channel: {PER_CHANNEL}")
    print(f"  argmax_in_model: True")
    print(f"  Model: {args.model}")
    print(f"  Output: {out_path}")
    print("=" * 70)

    # Load only lm_head weight
    print("\nLoading lm_head weight only...")
    t0 = time.time()
    weight = load_lm_head_weight(args.model, hidden_size, vocab_size)
    print(f"  Weight shape: {weight.shape}, dtype: {weight.dtype}")
    print(f"  Loaded in {time.time()-t0:.1f}s ({weight.numel()*2/1e6:.0f} MB)")

    # Create wrapper directly with loaded weight (no double allocation)
    wrapper = LMHeadWrapper(weight, argmax_mode=True).eval()
    del weight; gc.collect()

    # Trace
    print("\nTracing model...")
    sample_input = torch.zeros((1, 1, hidden_size), dtype=MODEL_DTYPE)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample_input)

    # Convert to CoreML (fp16 first, then quantize separately to save memory)
    # Use EMPTY pass pipeline to avoid OOM on 16GB machines with 248K vocab
    print("Converting to CoreML (fp16, minimal pipeline)...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_states", shape=sample_input.shape, dtype=np.float16)],
        outputs=[
            ct.TensorType(name="argmax_idx", dtype=np.int32),
            ct.TensorType(name="argmax_val", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
        pass_pipeline=ct.PassPipeline.EMPTY,
    )
    del traced, wrapper; gc.collect()
    print(f"  Converted in {time.time()-t0:.1f}s")

    # Save fp16 intermediate to disk, then free memory before quantization
    fp16_path = os.path.join(args.output, "_lm_head_fp16_argmax_tmp.mlpackage")
    if os.path.exists(fp16_path):
        shutil.rmtree(fp16_path)
    mlmodel.save(fp16_path)
    del mlmodel; gc.collect()
    print(f"  Saved fp16 intermediate → {fp16_path}")

    # Reload and apply LUT6
    print(f"\nReloading and applying LUT{LM_HEAD_LUT} quantization (per_channel={PER_CHANNEL})...")
    t0 = time.time()
    mlmodel = ct.models.MLModel(fp16_path)
    mlmodel = palettize_model(mlmodel, LM_HEAD_LUT, PER_CHANNEL)
    print(f"  Quantized in {time.time()-t0:.1f}s")

    # Save
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    mlmodel.save(out_path)
    del mlmodel; gc.collect()

    # Print size
    total_bytes = 0
    for dirpath, _, filenames in os.walk(out_path):
        for fn in filenames:
            total_bytes += os.path.getsize(os.path.join(dirpath, fn))
    print(f"\n  Saved lm_head ({total_bytes/1e6:.1f} MB)")

    # Quick load test
    print("\nQuick load test (CPU_AND_NE)...")
    try:
        loaded = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        spec = loaded.get_spec()
        output_names = [o.name for o in spec.description.output]
        print(f"  Outputs: {output_names}")
        assert "argmax_idx" in output_names, f"Expected argmax_idx, got {output_names}"
        assert "argmax_val" in output_names, f"Expected argmax_val, got {output_names}"

        dummy = np.zeros((1, 1, hidden_size), dtype=np.float16)
        result = loaded.predict({"hidden_states": dummy})
        print(f"  argmax_idx shape: {result['argmax_idx'].shape}, dtype: {result['argmax_idx'].dtype}")
        print(f"  argmax_val shape: {result['argmax_val'].shape}, dtype: {result['argmax_val'].dtype}")
        print("  Load test PASSED")
        del loaded
    except Exception as e:
        print(f"  Load test FAILED: {e}")

    print(f"\n{'='*70}")
    print("  DONE — lm_head.mlpackage exported with LUT6 + argmax")
    print(f"{'='*70}")

    # Clean up temp file
    if os.path.exists(fp16_path):
        shutil.rmtree(fp16_path)
        print("  Cleaned up fp16 intermediate")


if __name__ == "__main__":
    main()
